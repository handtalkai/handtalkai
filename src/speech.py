"""
speech.py - Reliable Windows TTS using subprocess (most stable approach)
Spawns a fresh PowerShell process for each sentence — avoids pyttsx3
engine death after first call, which is the root cause of the bug.
Falls back to pyttsx3 if PowerShell is not available.
"""

import subprocess
import threading
import queue
import time
import sys

_voice_cache = None


def list_voices():
    """
    Enumerate installed system voices via pyttsx3. Works regardless of
    which engine actually speaks (PowerShell or pyttsx3) - on Windows
    both sit on top of the same SAPI5 voice pool, so a voice name found
    here (e.g. "Microsoft David Desktop") is valid for either.

    Returns [{"id": <pyttsx3 voice id>, "name": <friendly name>}, ...],
    or [] if pyttsx3/voices aren't available. Cached after first call -
    installed voices don't change while the server is running.
    """
    global _voice_cache
    if _voice_cache is not None:
        return _voice_cache

    try:
        import pyttsx3
        engine = pyttsx3.init()
        _voice_cache = [
            {"id": v.id, "name": v.name} for v in engine.getProperty("voices")
        ]
        engine.stop()
    except Exception as e:
        print(f"[Speech] Could not enumerate voices: {e}")
        _voice_cache = []

    return _voice_cache


class SpeechOutput:
    """
    Speaks full sentences reliably on Windows.
    Uses PowerShell SpeechSynthesizer — no engine reuse bugs,
    no threading issues, works every time.
    """

    def __init__(self, rate: int = 130):
        """
        rate: words per minute. 130 = slow and clear.
              Normal speech ~150, fast ~200.
        """
        self._base_rate  = rate
        self._speed      = 1.0    # multiplier applied to _base_rate, from Settings' Speech Speed (0.5-2.0)
        self._volume     = 100    # 0-100, from Settings' Speech Volume
        self._voice_name = None   # None = system default; else a name from list_voices()
        self._queue  = queue.Queue()
        self._method = self._detect_method()

        print(f"[Speech] TTS initialized via {self._method} | rate={rate} wpm")

        worker = threading.Thread(target=self._worker, daemon=True)
        worker.start()

    def set_speed(self, multiplier):
        """Update the speaking rate multiplier (0.5x-2.0x). Applies to the
        next speak() call - matches the Settings page's Speech Speed slider."""
        try:
            self._speed = max(0.5, min(2.0, float(multiplier)))
        except (TypeError, ValueError):
            pass

    def set_volume(self, volume):
        """Update the speaking volume (0-100). Applies to the next speak()
        call - matches the Settings page's Speech Volume slider."""
        try:
            self._volume = max(0, min(100, int(volume)))
        except (TypeError, ValueError):
            pass

    def set_voice(self, name):
        """Update the selected voice by its friendly name (Settings page's
        Speech Voice dropdown). Falls back to the system default (None) if
        name is empty or isn't one of the currently installed voices -
        e.g. a voice saved on one machine that isn't installed on this one."""
        if not name:
            self._voice_name = None
            return
        match = next((v for v in list_voices() if v["name"] == name), None)
        self._voice_name = match["name"] if match else None

    def _detect_method(self) -> str:
        """Detect the best available TTS method."""
        if sys.platform == "win32":
            # Test if PowerShell is available
            try:
                result = subprocess.run(
                    ["powershell", "-Command", "echo ok"],
                    capture_output=True, timeout=3
                )
                if result.returncode == 0:
                    return "powershell"
            except Exception:
                pass

        # Fallback: pyttsx3
        try:
            import pyttsx3
            return "pyttsx3"
        except ImportError:
            pass

        return "none"

    def speak(self, text: str):
        """
        Speak the full sentence.
        Clears any pending speech so new sentence plays immediately.
        """
        if self._method == "none" or not text.strip():
            return

        # Clear pending items
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

        print(f"[Speech] Speaking: '{text}'")
        self._queue.put(text)

    def _worker(self):
        """Background thread — speaks one sentence at a time."""
        while True:
            text = self._queue.get()
            if text is None:
                break
            try:
                if self._method == "powershell":
                    self._speak_powershell(text)
                elif self._method == "pyttsx3":
                    self._speak_pyttsx3(text)
            except Exception as e:
                print(f"[Speech] Error: {e}")

    def _speak_powershell(self, text: str):
        """
        Use Windows built-in SpeechSynthesizer via PowerShell.
        A fresh process each time = no state corruption between sentences.
        """
        # Escape single quotes in text
        safe = text.replace("'", " ")
        wpm  = self._base_rate * self._speed
        rate = wpm - 150   # PowerShell rate: -10 slow, 0 normal, 10 fast
        rate = max(-10, min(10, int(rate // 15)))
        volume = max(0, min(100, self._volume))

        # pyttsx3's voice name includes a " - <language>" suffix that
        # System.Speech.Synthesis.SpeechSynthesizer doesn't use - e.g.
        # pyttsx3 reports "Microsoft David Desktop - English (United
        # States)" while GetInstalledVoices() calls the same voice just
        # "Microsoft David Desktop". SelectVoice() requires an exact
        # match, so match by prefix here instead of passing our name
        # straight through (which would always throw and silently fall
        # back to the default voice via the try/catch).
        select_voice = ""
        if self._voice_name:
            safe_voice = self._voice_name.replace("'", " ")
            select_voice = (
                f"$vi = $s.GetInstalledVoices() | Where-Object {{ '{safe_voice}'.StartsWith($_.VoiceInfo.Name) }} | Select-Object -First 1; "
                f"if ($vi) {{ try {{ $s.SelectVoice($vi.VoiceInfo.Name) }} catch {{}} }}; "
            )

        cmd = (
            f"Add-Type -AssemblyName System.Speech; "
            f"$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.Rate = {rate}; "
            f"$s.Volume = {volume}; "
            f"{select_voice}"
            f"$s.Speak('{safe}'); "
            f"$s.Dispose()"
        )
        subprocess.run(
            ["powershell", "-Command", cmd],
            capture_output=True,
            timeout=30,
        )

    def _speak_pyttsx3(self, text: str):
        """
        Fresh pyttsx3 engine per sentence — avoids the 'speaks once' bug
        by never reusing the same engine instance.
        """
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", self._base_rate * self._speed)
        engine.setProperty("volume", max(0, min(100, self._volume)) / 100)
        if self._voice_name:
            match = next(
                (v for v in engine.getProperty("voices") if v.name == self._voice_name),
                None,
            )
            if match:
                engine.setProperty("voice", match.id)
        engine.say(text)
        engine.runAndWait()
        engine.stop()

    def stop(self):
        self._queue.put(None)