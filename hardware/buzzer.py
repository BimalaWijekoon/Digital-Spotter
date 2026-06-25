"""
hardware/buzzer.py
Purpose: Active buzzer control via GPIO — sounds on bad form detection.
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

# Attempt GPIO import — only available on Raspberry Pi
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    logger.info("RPi.GPIO not available — buzzer in mock mode")


class Buzzer:
    """Active buzzer controller using GPIO.

    Drives an active buzzer module connected to a GPIO pin.
    Active buzzers only need HIGH/LOW — no PWM frequency control needed.
    Falls back to silent mock mode on non-Pi platforms.

    Attributes:
        _pin: GPIO BCM pin number.
        _enabled: Whether buzzer is enabled in config.
        _is_mock: True if GPIO not available.
    """

    def __init__(self, pin=None, enabled=None):
        """Initialize buzzer.

        Args:
            pin: GPIO BCM pin number. Defaults to Config.BUZZER.GPIO_PIN.
            enabled: Override enable flag. Defaults to Config.BUZZER.ENABLED.
        """
        self._pin = pin if pin is not None else Config.BUZZER.GPIO_PIN
        self._enabled = enabled if enabled is not None else Config.BUZZER.ENABLED
        self._is_mock = not GPIO_AVAILABLE
        self._is_setup = False
        self._lock = threading.Lock()

        if not self._is_mock and self._enabled:
            self._setup_gpio()

    def _setup_gpio(self):
        """Configure GPIO pin as output.

        Side Effects:
            Sets GPIO mode and pin direction.
        """
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self._pin, GPIO.OUT, initial=GPIO.LOW)
            self._is_setup = True
            logger.info("Buzzer initialized on GPIO %d", self._pin)
        except Exception as e:
            logger.error("Failed to setup GPIO %d: %s", self._pin, e)
            self._is_mock = True

    def beep_bad_form(self):
        """Sound the triple rapid beep pattern for bad form detection.

        Non-blocking — runs in a background thread so inference
        pipeline is not delayed.

        Side Effects:
            Fires GPIO pin HIGH/LOW in the configured pattern.
        """
        if not self._enabled:
            return

        pattern = Config.BUZZER.BAD_FORM_PATTERN
        threading.Thread(
            target=self._play_pattern,
            args=(pattern,),
            daemon=True,
        ).start()

    def beep_test(self):
        """Single short beep for testing buzzer hardware.

        Non-blocking — runs in a background thread.

        Side Effects:
            Fires GPIO pin HIGH for 150ms.
        """
        threading.Thread(
            target=self._play_pattern,
            args=([0.15],),
            daemon=True,
        ).start()

    def _play_pattern(self, pattern):
        """Play a beep pattern: alternating on/off durations.

        Args:
            pattern: List of durations in seconds.
                Odd indices (0, 2, 4...) = buzzer ON.
                Even indices (1, 3...) = buzzer OFF (silence gap).

        Side Effects:
            Drives GPIO pin HIGH/LOW.
        """
        with self._lock:
            for i, duration in enumerate(pattern):
                is_on = (i % 2 == 0)
                if is_on:
                    self._set_pin(True)
                else:
                    self._set_pin(False)
                time.sleep(duration)
            # Always end LOW
            self._set_pin(False)

    def _set_pin(self, high):
        """Set GPIO pin state.

        Args:
            high: True for HIGH, False for LOW.

        Side Effects:
            Writes to GPIO pin.
        """
        if self._is_mock:
            state = "ON" if high else "OFF"
            logger.debug("Buzzer MOCK: %s", state)
            return

        if self._is_setup:
            try:
                GPIO.output(self._pin, GPIO.HIGH if high else GPIO.LOW)
            except Exception as e:
                logger.error("GPIO output error: %s", e)

    @property
    def is_mock(self):
        """Check if using mock (no-op) mode.
        Returns:
            bool: True if GPIO not available.
        """
        return self._is_mock

    @property
    def is_enabled(self):
        """Check if buzzer is enabled in config.
        Returns:
            bool: True if enabled.
        """
        return self._enabled

    def cleanup(self):
        """Release GPIO resources.

        Side Effects:
            Calls GPIO.cleanup() for the buzzer pin.
        """
        if not self._is_mock and self._is_setup:
            try:
                GPIO.output(self._pin, GPIO.LOW)
                GPIO.cleanup(self._pin)
                logger.info("Buzzer GPIO %d cleaned up", self._pin)
            except Exception as e:
                logger.error("GPIO cleanup error: %s", e)

    def __del__(self):
        """Ensure GPIO is released on garbage collection."""
        self.cleanup()
