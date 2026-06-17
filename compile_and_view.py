import sys
import os
import io
import tarfile
import requests
import webbrowser
from PIL import Image, ImageDraw

def generate_hust_logo():
    logo_path = "hust_logo.png"
    if os.path.exists(logo_path):
        print("[*] Logo hust_logo.png already exists. Skipping generation.")
        return
    print("[*] Generating HUST placeholder logo...")
    # Create red background (HUST color)
    img = Image.new('RGB', (200, 200), color=(180, 0, 0))
    draw = ImageDraw.Draw(img)
    # Draw double gold borders
    draw.rectangle([10, 10, 190, 190], outline=(255, 215, 0), width=4)
    draw.rectangle([18, 18, 182, 182], outline=(255, 215, 0), width=2)
    # Draw star shape inside
    draw.polygon([
        (100, 35), (117, 74), (159, 74), (125, 98), 
        (138, 137), (100, 113), (62, 137), (75, 98), 
        (41, 74), (83, 74)
    ], fill=(255, 215, 0))
    img.save(logo_path)
    print("[+] Logo saved as hust_logo.png.")

def compile_latex():
    tex_file = "report.tex"
    logo_file = "hust_logo.png"
    
    if not os.path.exists(tex_file):
        print(f"[-] Error: {tex_file} not found!")
        sys.exit(1)
        
    generate_hust_logo()
    
    print("\n[*] Preparing files for online compilation...")
    tar_stream = io.BytesIO()
    with tarfile.open(fileobj=tar_stream, mode='w:bz2') as tar:
        # Add report.tex
        tar.add(tex_file, arcname="report.tex")
        # Add hust_logo.png
        tar.add(logo_file, arcname="hust_logo.png")
        
        # Add WHU-LX reproduction figures
        whulx_figs = [
            "artifacts/outputs/whulx_reproduction/training_history_plot.png",
            "artifacts/outputs/whulx_reproduction/controller_comparison_metrics.png",
            "artifacts/outputs/whulx_reproduction/performance_comparison.png",
            "artifacts/outputs/whulx_reproduction/controller_total_reward.png",
            "artifacts/outputs/hcm_summer/cumulative_energy.png",
            "artifacts/outputs/hcm_summer/energy_comfort_tradeoff.png",
        ]
        for fig in whulx_figs:
            if os.path.exists(fig):
                tar.add(fig, arcname=fig)
                print(f"[+] Added figure: {fig}")
            else:
                print(f"[!] Warning: figure {fig} not found on disk!")
    tar_stream.seek(0)
    
    print("[*] Sending compilation request to latexonline.cc...")
    print("[*] This process may take 15-45 seconds depending on server load...")
    url = "https://latexonline.cc/data?target=report.tex&command=pdflatex"
    try:
        response = requests.post(
            url,
            files={"file": ("report.tar.bz2", tar_stream, "application/x-bzip2")}
        )
        if response.status_code == 200:
            pdf_path = "report.pdf"
            with open(pdf_path, "wb") as f:
                f.write(response.content)
            print(f"\n==========================================")
            print(f" SUCCESS: PDF compiled successfully!")
            print(f" Output saved to: {os.path.abspath(pdf_path)}")
            print(f"==========================================\n")
            print("[*] Opening PDF in default browser/viewer...")
            webbrowser.open(pdf_path)
        else:
            print(f"\n==========================================")
            print(f" ERROR: Compilation failed (HTTP {response.status_code})")
            print(f"==========================================\n")
            log_path = "report.log"
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(response.text)
            print(f"LaTeX Compiler Log saved to: {os.path.abspath(log_path)}")
            print("\nCompiler Output Log Snippet:")
            try:
                # Reconfigure stdout if possible
                if hasattr(sys.stdout, 'reconfigure'):
                    sys.stdout.reconfigure(encoding='utf-8')
                print(response.text)
            except Exception:
                try:
                    encoding = sys.stdout.encoding or 'utf-8'
                    print(response.text.encode(encoding, errors='replace').decode(encoding))
                except Exception:
                    print("Could not display logs due to console encoding limits. Please open report.log to view details.")
    except Exception as e:
        print(f"[-] An error occurred during communication: {e}")

if __name__ == "__main__":
    compile_latex()
