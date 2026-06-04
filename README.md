# HVAC XGB-DQN Simulation Starter

Bo khung nay dung de khoi dong du an mo phong HVAC + cua so theo huong XGB-DQN.

## Muc tieu

- Dung repo goc `WHU-LX/Hvac-Window-based-XGB-DQN` lam baseline hoc thuat.
- Dung du lieu va pipeline mo rong tu `trandat09062003/Hvac-Window-based-XGB-DQN`.
- Chay va giai thich bang notebook, khong bat dau tu mot bo scripts rieng cua nhom.

## Cau truc

- `config/project_paths.json`: Cau hinh duong dan baseline, repo mo rong, file CSV, file EPW va EnergyPlus.
- `data/`: Chua `Cleaned_data.csv` va `Cleaned_data_encode.csv` lay tu WHU-LX de tai hien XGB-DQN ngay trong repo.
- `artifacts/`: Noi sinh ra models, outputs, generated_data khi chay.
- `notebooks/01_bootstrap_hanoi_simulation.ipynb`: Notebook bootstrap cho huong Sinergym/EnergyPlus.
- `notebooks/02_reproduce_whulx_pipeline.ipynb`: Notebook tai hien pipeline WHU-LX XGB-DQN tu `scripts/reproduce_whulx_pipeline.py`.
- `requirements.txt`: Phu thuoc de mo notebook va chay mo phong.

## Nguon tham chieu

- Baseline: `../references/Hvac-Window-based-XGB-DQN`
- Mo rong: `../references/Hvac-Window-based-XGB-DQN-extended`

## Cach dung

1. Tao moi truong Python va cai `requirements.txt`.
2. De tai hien baseline WHU-LX, mo `notebooks/02_reproduce_whulx_pipeline.ipynb` va chay tung cell. Notebook doc du lieu truc tiep tu `data/Cleaned_data.csv`.
3. De chay huong mo phong Sinergym/EnergyPlus, mo `config/project_paths.json` va sua `eplus_path` theo thu muc EnergyPlus tren may, sau do mo `notebooks/01_bootstrap_hanoi_simulation.ipynb`.
4. Chay tung cell trong notebook bootstrap:
   - kiem tra duong dan va du lieu,
   - sinh du lieu mo phong Sinergym,
   - train agent DQN,
   - evaluate va luu ket qua vao `artifacts/`.

## Ghi chu

- Repo mo rong hien tai dung cac scripts co san trong `trandat09062003/Hvac-Window-based-XGB-DQN`.
- Notebook nay dong vai tro "runbook" de nhom giai thich tung buoc va chay thu nghiem.
