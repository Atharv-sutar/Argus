import urllib.request

def test_urls():
    urls = {
        "osnet_x0_25_msmt17": "https://huggingface.co/paulosantiago/osnet_x0_25_msmt17/resolve/main/osnet_x0_25_msmt17.pt",
        "libre_x1_0_msmt17": "https://huggingface.co/LibreYOLO/LibreReID-osnet/resolve/main/osnet_x1_0_msmt17.pt",
        "libre_x1_0_market": "https://huggingface.co/LibreYOLO/LibreReID-osnet/resolve/main/osnet_x1_0_market1501.pt",
        "libre_ain_x1_0_msmt17": "https://huggingface.co/LibreYOLO/LibreReID-osnet/resolve/main/osnet_ain_x1_0_msmt17.pt",
        "libre_ain_x1_0_market": "https://huggingface.co/LibreYOLO/LibreReID-osnet/resolve/main/osnet_ain_x1_0_market1501.pt",
    }
    for name, url in urls.items():
        try:
            req = urllib.request.Request(url, method='HEAD')
            with urllib.request.urlopen(req, timeout=5) as resp:
                print(f"[OK] {name} -> {resp.status} (Length: {resp.headers.get('Content-Length')})")
        except Exception as e:
            print(f"[FAIL] {name} -> {e}")

if __name__ == "__main__":
    test_urls()
