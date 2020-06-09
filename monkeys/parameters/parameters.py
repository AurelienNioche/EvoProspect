import os

FIG_FOLDER = 'fig'
BACKUP_FOLDER = os.path.join('data', 'pickle')
SOURCE_FOLDER = os.path.join('data', 'source')
EXPORT_FOLDER = os.path.join('data', 'export_xlsx')
print(f"The figure folder is: {os.path.abspath(FIG_FOLDER)}")
print(f"The backup folder is: {os.path.abspath(BACKUP_FOLDER)}")
print(f"The source folder is: {os.path.abspath(SOURCE_FOLDER)}")
print(f"The export folder is: {os.path.abspath(EXPORT_FOLDER)}")
print()

# Create folders
FOLDERS = FIG_FOLDER, BACKUP_FOLDER, SOURCE_FOLDER, EXPORT_FOLDER
for folder in FOLDERS:
    os.makedirs(folder, exist_ok=True)

SAME_P = 'same p'
SAME_X = 'same x'

LABELS_CONTROL = {
    SAME_P: "Diff. $x$, same $p$",
    SAME_X: "Diff. $p$, same $x$"
}

CONTROL_CONDITIONS = SAME_P, SAME_X

SIG_STEEP = 'sig_steep'
SIG_MID = 'sig_mid'

GAIN = "gain"
LOSS = "loss"
