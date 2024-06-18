# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(
    ['main.py','classify.py','featureconfigwindow.py','featureselection.py','log.py','mainprocedure.py','preprocessing.py','ui.py','utils.py'],
    pathex=[],
    binaries=[],    datas=[('./radiomicstool.ico','./radiomicstool.ico'),('./yaml/Params.yaml','./yaml/Params.yaml'),('xgboost','xgboost')],    hiddenimports=['radiomics.base','radiomics.featureextractor','radiomics.firstorder','radiomics.generalinfo','radiomics.glcm','radiomics.gldm','radiomics.glrlm','radiomics.glszm','radiomics.imageoperations','radiomics.ngtdm','radiomics.shape','radiomics.shape2D'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='radiomicstool',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['radiomicstool.ico'],
)
