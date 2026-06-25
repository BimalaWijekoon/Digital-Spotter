"""
hardware/rgb_led.py
Purpose: RGB 5050 SMD LED control via GPIO PWM — visual form feedback.
         Green = good form, red = bad form, amber = processing.
         Falls back to mock (no-op) mode on non-Pi platforms.
Author: bimalawijekoon
Version: 1.0.0
Last Modified: 2026-06-25
"""

import logging
import threading
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import Config

logger = logging.getLogger(__name__)

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    logger.info("RPi.GPIO not available — RGB LED in mock mode")


class RgbLed:
    """RGB 5050 SMD LED controller using GPIO PWM.

    Drives a common-cathode or common-anode RGB LED module via
    3 PWM-capable GPIO pins. Falls back to mock mode on non-Pi.

    Colors are specified as (R, G, B) tuples with values 0–255.
    """

    # Predefined colors
    OFF     = (0, 0, 0)
    RED     = (255, 0, 0)
    GREEN   = (0, 255, 0)
    BLUE    = (0, 0, 255)
    AMBER   = (255, 100, 0)
    CYAN    = (0, 255, 255)
    WHITE   = (255, 255, 255)
    PURPLE  = (180, 0, 255)

    def __init__(self, pin_r=None, pin_g=None, pin_b=None,
                 common_anode=False, enabled=None):
        """Initialize RGB LED.

        Args:
            pin_r: GPIO BCM pin for Red channel. Defaults to Config.
            pin_g: GPIO BCM pin for Green channel. Defaults to Config.
            pin_b: GPIO BCM pin for Blue channel. Defaults to Config.
            common_anode: True if module is common-anode (inverted logic).
            enabled: Override enable flag. Defaults to Config.RGB_LED.ENABLED.
        """
        self._pin_r = pin_r if pin_r is not None else Config.RGB_LED.PIN_R
        self._pin_g = pin_g if pin_g is not None else Config.RGB_LED.PIN_G
        self._pin_b = pin_b if pin_b is not None else Config.RGB_LED.PIN_B
        self._common_anode = common_anode
        self._enabled = enabled if enabled is not None else Config.RGB_LED.ENABLED
        self._is_mock = not GPIO_AVAILABLE
        self._is_setup = False
        self._pwm_r = None
        self._pwm_g = None
        self._pwm_b = None
        self._lock = threading.Lock()
        self._flash_thread = None
        self._flash_stop = threading.Event()

        if not self._is_mock and self._enabled:
            self._setup_gpio()

    def _setup_gpio(self):
        """Configure GPIO pins with PWM at 1kHz.

        Side Effects:
            Sets GPIO mode, configures pins as PWM outputs.
        """
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self._pin_r, GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(self._pin_g, GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(self._pin_b, GPIO.OUT, initial=GPIO.LOW)

            self._pwm_r = GPIO.PWM(self._pin_r, 1000)
            self._pwm_g = GPIO.PWM(self._pin_g, 1000)
            self._pwm_b = GPIO.PWM(self._pin_b, 1000)

            self._pwm_r.start(0)
            self._pwm_g.start(0)
            self._pwm_b.start(0)

            self._is_setup = True
            logger.info("RGB LED initialized on GPIO R=%d G=%d B=%d",
                        self._pin_r, self._pin_g, self._pin_b)
        except Exception as e:
            logger.error("Failed to setup RGB LED: %s", e)
            self._is_mock = True

    def set_color(self, rgb):
        """Set the LED to a solid color.

        Args:
            rgb: Tuple of (R, G, B) with values 0–255.

        Side Effects:
            Changes PWM duty cycle on all 3 channels.
        """
        if not self._enabled:
            return

        self._stop_flash()
        self._apply_color(rgb)

    def _apply_color(self, rgb):
        """Low-level color application.

        Args:
            rgb: Tuple of (R, G, B) with values 0–255.
        """
        r, g, b = rgb

        # Convert 0–255 to 0–100 duty cycle
        dr = (r / 255.0) * 100.0
        dg = (g / 255.0) * 100.0
        db = (b / 255.0) * 100.0

        # Invert for common-anode modules
        if self._common_anode:
            dr = 100.0 - dr
            dg = 100.0 - dg
            db = 100.0 - db

        if self._is_mock:
            logger.debug("RGB MOCK: (%d, %d, %d) → duty (%.0f, %.0f, %.0f)",
                         r, g, b, dr, dg, db)
            return

        if self._is_setup:
            try:
                self._pwm_r.ChangeDutyCycle(dr)
                self._pwm_g.ChangeDutyCycle(dg)
                self._pwm_b.ChangeDutyCycle(db)
            except Exception as e:
                logger.error("RGB PWM error: %s", e)

    def good_form(self):
        """Set LED to solid green — good form detected.

        Side Effects:
            Sets LED color, stops any active flash.
        """
        self.set_color(self.GREEN)

    def bad_form(self):
        """Flash red — bad form detected.

        Non-blocking — flashes 3 times in a background thread,
        then holds solid red for 2 seconds before turning off.

        Side Effects:
            Starts flash thread.
        """
        if not self._enabled:
            return
        self._stop_flash()
        self._flash_stop.clear()
        self._flash_thread = threading.Thread(
            target=self._flash_pattern,
            args=(self.RED, 3, 0.15, 0.1),
            daemon=True,
        )
        self._flash_thread.start()

    def processing(self):
        """Set LED to amber — inference in progress.

        Side Effects:
            Sets LED color, stops any active flash.
        """
        self.set_color(self.AMBER)

    def idle(self):
        """Set LED to dim blue — system idle/standby.

        Side Effects:
            Sets LED to low-brightness blue.
        """
        self.set_color((0, 0, 30))

    def off(self):
        """Turn LED off.

        Side Effects:
            Sets all channels to 0.
        """
        self.set_color(self.OFF)

    def _flash_pattern(self, color, count, on_time, off_time):
        """Flash a color N times, then hold for 2s, then off.

        Args:
            color: (R, G, B) tuple.
            count: Number of flashes.
            on_time: Seconds LED is on per flash.
            off_time: Seconds LED is off between flashes.

        Side Effects:
            Drives GPIO PWM in a timed loop.
        """
        for i in range(count):
            if self._flash_stop.is_set():
                break
            self._apply_color(color)
            time.sleep(on_time)
            self._apply_color(self.OFF)
            time.sleep(off_time)

        if not self._flash_stop.is_set():
            # Hold solid for 2 seconds after flashing
            self._apply_color(color)
            time.sleep(2.0)
            self._apply_color(self.OFF)

    def _stop_flash(self):
        """Stop any active flash thread.

        Side Effects:
            Sets stop event and waits for thread to finish.
        """
        self._flash_stop.set()
        if self._flash_thread and self._flash_thread.is_alive():
            self._flash_thread.join(timeout=1.0)

    @property
    def is_mock(self):
        """Check if using mock mode.
        Returns:
            bool: True if GPIO not available.
        """
        return self._is_mock

    @property
    def is_enabled(self):
        """Check if LED is enabled in config.
        Returns:
            bool: True if enabled.
        """
        return self._enabled

    def cleanup(self):
        """Release GPIO PWM resources.

        Side Effects:
            Stops PWM and calls GPIO.cleanup for LED pins.
        """
        self._stop_flash()
        if not self._is_mock and self._is_setup:
            try:
                self._pwm_r.stop()
                self._pwm_g.stop()
                self._pwm_b.stop()
                GPIO.cleanup([self._pin_r, self._pin_g, self._pin_b])
                logger.info("RGB LED GPIO cleaned up")
            except Exception as e:
                logger.error("RGB cleanup error: %s", e)

    def __del__(self):
        """Ensure GPIO released on garbage collection."""
        self.cleanup()
