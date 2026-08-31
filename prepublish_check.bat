@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo Pre-publication repository check
echo ============================================================
echo.

python -c ^
"from pathlib import Path; ^
root=Path('.'); ^
exts={'.py','.md','.txt','.bat','.json','.csv'}; ^
skip={'__pycache__','.git','.venv','venv','env'}; ^
patterns=['sk-'+'proj-','AI'+'za','sk-'+'ant-']; ^
print('[1] Searching for obvious API-key patterns...'); ^
hits=[]; ^
[(hits.append((str(p),i,line.strip()))) ^
 for p in root.rglob('*') ^
 if p.is_file() and p.suffix.lower() in exts ^
 and not any(x in skip for x in p.parts) ^
 and p.name.lower()!='prepublish_check.bat' ^
 for i,line in enumerate(p.read_text(encoding='utf-8',errors='ignore').splitlines(),1) ^
 if any(q in line for q in patterns)]; ^
[print(f'  {p}:{i}: {line[:180]}') for p,i,line in hits]; ^
print('  [OK] No obvious API keys found.' if not hits else f'  [WARNING] {len(hits)} possible secret(s) found.'); ^
print(); ^
print('[2] Searching for machine-specific absolute paths...'); ^
paths=['C:'+chr(92)+'Users'+chr(92),'D:'+chr(92)+'download'+chr(92),'D:'+chr(92)+'DOWNLOAD'+chr(92)]; ^
hits2=[]; ^
[(hits2.append((str(p),i,line.strip()))) ^
 for p in root.rglob('*') ^
 if p.is_file() and p.suffix.lower() in exts ^
 and not any(x in skip for x in p.parts) ^
 and p.name.lower()!='prepublish_check.bat' ^
 for i,line in enumerate(p.read_text(encoding='utf-8',errors='ignore').splitlines(),1) ^
 if any(q in line for q in paths)]; ^
[print(f'  {p}:{i}: {line[:180]}') for p,i,line in hits2]; ^
print('  [OK] No machine-specific absolute paths found.' if not hits2 else f'  [WARNING] {len(hits2)} absolute path(s) found.')"

echo.
echo [3] Searching for files that should normally not be committed...

for /r %%F in (*.pt *.pth *.ckpt *.npz *.npy *.pyc) do (
    echo   [WARNING] %%F
)

echo.
echo [4] Checking required repository files...

if exist ".gitignore" (
    echo   [OK] .gitignore
) else (
    echo   [WARNING] .gitignore missing
)

if exist "README.md" (
    echo   [OK] README.md
) else (
    echo   [WARNING] README.md missing
)

if exist "requirements.txt" (
    echo   [OK] requirements.txt
) else (
    echo   [WARNING] requirements.txt missing
)

echo.
echo ============================================================
echo Check complete.
echo ============================================================
pause