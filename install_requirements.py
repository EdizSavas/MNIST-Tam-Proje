import subprocess
import sys

packages = [
    "PyQt6",
    "numpy",
    "scipy",
    "matplotlib",
    "faster-whisper",
    "pyautogui",
    "rapidfuzz",
    "pyaudio",
    "keyboard",
    "psutil",
    "sounddevice",
    "selenium"
    # "openai-whisper",  # GEREK YOK: faster-whisper yetiyor
]

def install(package):
    print(f"{package} [Yükleniyor...]")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def main():
    print("[+] Gerekli kütüphaneler yükleniyor...\n")
    for pkg in packages:
        try:
            install(pkg)
        except Exception as e:
            print(f"[HATA] {pkg} kurulamadı: {e}")
    print("\n[+] Yükleme tamamlandı.")

if __name__ == "__main__":
    main()
