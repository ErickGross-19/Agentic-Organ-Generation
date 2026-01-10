# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Organ Generator executable.

Build with:
    pyinstaller build.spec

Or for a single-file executable:
    pyinstaller build.spec --onefile
"""

import sys
from pathlib import Path

block_cipher = None

# Get the project root directory
project_root = Path(SPECPATH)

# Collect data files
datas = [
    # Include automation module
    (str(project_root / 'automation'), 'automation'),
    # Include generation module
    (str(project_root / 'generation'), 'generation'),
    # Include validity module
    (str(project_root / 'validity'), 'validity'),
    # Include GUI module
    (str(project_root / 'gui'), 'gui'),
]

# Hidden imports that PyInstaller might miss
hiddenimports = [
    # Tkinter
    'tkinter',
    'tkinter.ttk',
    'tkinter.messagebox',
    'tkinter.filedialog',
    'tkinter.scrolledtext',
    # Matplotlib
    'matplotlib',
    'matplotlib.pyplot',
    'matplotlib.backends.backend_tkagg',
    'mpl_toolkits.mplot3d',
    # Scientific computing
    'numpy',
    'scipy',
    'trimesh',
    'networkx',
    # Image processing
    'PIL',
    'skimage',
    'skimage.measure',
    # LLM clients
    'openai',
    'anthropic',
    # Project modules
    'automation',
    'automation.workflow',
    'automation.agent_runner',
    'automation.llm_client',
    'generation',
    'generation.api',
    'generation.core',
    'generation.ops',
    'generation.specs',
    'generation.adapters',
    'validity',
    'gui',
    'gui.main_window',
    'gui.workflow_manager',
    'gui.stl_viewer',
    'gui.agent_config',
    'gui.security',
]

a = Analysis(
    ['main.py'],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary modules to reduce size
        'test',
        'tests',
        'pytest',
        'pytest_cov',
        'sphinx',
        'IPython',
        'jupyter',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher,
)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='OrganGenerator',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # Set to True for debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon path here: 'icon.ico' for Windows, 'icon.icns' for macOS
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='OrganGenerator',
)

# For macOS app bundle (optional)
# app = BUNDLE(
#     coll,
#     name='OrganGenerator.app',
#     icon='icon.icns',
#     bundle_identifier='com.organgeneration.app',
#     info_plist={
#         'NSHighResolutionCapable': 'True',
#         'CFBundleShortVersionString': '1.0.0',
#     },
# )
