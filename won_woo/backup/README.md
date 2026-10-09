# won_woo 서버 백업

백업 대상은 `/home/viglab2/won_woo`입니다. GitHub 저장소는 `https://github.com/wonulee/PRVR-develop.git`, 백업 브랜치는 `backup/won-woo-20261009`, 코드 스냅샷은 브랜치 안의 `won_woo/`입니다. 사용자가 공개 저장소 백업을 선택했습니다.

Google Drive 백업 폴더: https://drive.google.com/drive/folders/1ltsNV5c_kLf2eOjd-MhNc8pQ_1LExFsK

대용량 파일을 보관하는 `files` 폴더: https://drive.google.com/drive/folders/1s6O6tIQpTp2QdHeN-0IZd1zuvNsxoPU1

2026-10-09 현재 Drive 대용량 업로드와 전송 검증은 아직 완료되지 않았습니다. 폴더 생성이나 로컬 manifest 생성만으로 백업 완료를 판단하지 마세요.

## 파일 분리

2MiB 이하의 코드·설정·문서·작은 실험 기록은 GitHub에 보관합니다. Feature, checkpoint, 모델 가중치, 배열·압축파일 및 확장자 없는 바이너리는 크기와 무관하게 Google Drive에 보관합니다. 2MiB를 초과하는 CSV/TXT/JSONL 등도 Drive 대상입니다. `.pyc`, 임시 백업 작업 폴더와 자격증명 설정은 제외합니다. `manifest.json`은 파일의 목적지·크기·원래 경로, SHA256 및 심볼릭 링크를 기록합니다.

원본 학습 파일과 checkpoint는 백업 작업에서 수정하지 않습니다. 전체 약 166GiB를 복사하거나 하나의 큰 압축파일로 만들지 않고, Drive 파일 목록에 따라 직접 전송합니다.

## 목록과 GitHub 스냅샷 만들기

```bash
python3 backup/backup_workspace.py prepare
python3 backup/backup_workspace.py hash --target all
python3 backup/backup_workspace.py stage --output .backup_work/github_repository
```

이 순서로 실행해야 SHA256이 채워진 manifest를 GitHub 스냅샷에 포함합니다. `hash`는 파일별 결과를 저장하므로 같은 명령으로 이어서 실행할 수 있습니다. `prepare`를 다시 실행하면 SHA256이 초기화되므로 다시 `hash --target all`을 실행하세요.

`stage`는 원본을 수정하지 않고 GitHub용 파일만 `출력/won_woo/`로 복사합니다. 절대 심볼릭 링크는 스냅샷에서 상대 경로로 보관합니다. 지정한 스냅샷 폴더가 이미 있으면 새 경로를 사용하세요. manifest와 두 파일 목록은 자기 자신을 해시하는 순환을 피하기 위해 SHA256 검증 대상에서 제외합니다.

## Drive 대용량 전송

별도의 서버나 PC에서 대용량 전송을 실행하려면 rclone을 사용합니다. `rclone config`로 Google Drive remote `gdrive`를 설정하세요. 서버에 브라우저가 없으면 공식 remote setup 절차를 사용합니다. 인증 token과 rclone config는 GitHub에 넣지 마세요.

동명 폴더가 있어도 위 `files` 폴더를 정확히 사용하도록 folder ID를 지정합니다. `--drive-root-folder-id`는 해당 폴더를 remote의 루트로 설정하므로 목적지는 `gdrive:`입니다. 업로드 스크립트는 이 옵션을 `copy`와 후속 `check` 모두에 전달합니다.

```bash
bash backup/upload_drive.sh gdrive: --drive-root-folder-id 1s6O6tIQpTp2QdHeN-0IZd1zuvNsxoPU1 --dry-run
bash backup/upload_drive.sh gdrive: --drive-root-folder-id 1s6O6tIQpTp2QdHeN-0IZd1zuvNsxoPU1
```

매번 옵션을 지정하는 대신 `rclone config update gdrive root_folder_id 1s6O6tIQpTp2QdHeN-0IZd1zuvNsxoPU1`로 remote에 저장할 수도 있습니다. 이 설정은 이후 해당 remote를 사용하는 모든 명령에 적용됩니다.

전송은 `copy`를 사용해 대상의 기존 파일을 삭제하지 않습니다. 같은 명령을 다시 실행하면 이미 같은 파일은 건너뜁니다. 전송 뒤 `rclone check`로 비교하며 오류가 있으면 종료 상태가 실패입니다. `.backup_work/drive-upload.log`, `drive-check.log`를 확인하세요. 완료한 전송을 다시 실행하는 방식이며, 모든 업로드의 중간 바이트 위치 재개를 보장하지 않습니다.

## 복원

```bash
git clone --branch backup/won-woo-20261009 https://github.com/wonulee/PRVR-develop.git
cd PRVR-develop/won_woo
rclone copy gdrive: . --drive-root-folder-id 1s6O6tIQpTp2QdHeN-0IZd1zuvNsxoPU1 --files-from-raw backup/drive_files.txt --checksum --transfers 2
python3 backup/verify_restore.py --root "$PWD" --manifest backup/manifest.json
conda env create -f backup/environment/holmes.yml
conda run -n holmes python backup/restore_links.py --root "$PWD" --manifest backup/manifest.json
```

Drive에서 모든 파일을 내려받은 다음 `verify_restore.py`가 정규 파일의 SHA256·크기와 심볼릭 링크를 검사합니다. 누락 파일, 잘못된 해시 또는 끊어진 링크가 있으면 실패합니다. 이 검사는 `restore_links.py`보다 먼저 실행하세요. HDF5 링크를 수정한 뒤에는 원래 파일과 바이트가 달라질 수 있습니다.

HDF5의 ExternalLink는 파일 이름만 복원해도 데이터가 읽히지 않을 수 있습니다. 링크 복구 도구를 실행하고 외부 대상 feature도 모두 복원해야 합니다. 기존 JSON·실험 설정에 저장된 절대 경로는 분석 코드가 직접 읽을 수 있으므로, 가능하면 원래 `/home/viglab2/won_woo` 위치에 복원하거나 실행 설정의 경로를 조정하세요. 원본 감사 JSON을 일괄 치환하지 않습니다.

원래 경로에 의도적으로 복원한 경우 링크 복구 도구의 원본 보호 검사를 명시적으로 해제해야 합니다. 복원본에서만 다음 명령을 사용하세요.

```bash
conda run -n holmes python backup/restore_links.py --root "$PWD" --manifest backup/manifest.json --allow-source-root
```

서버 외부에 있던 Qwen 및 문장 모델은 `backup/environment/runtime-dependencies.json`에 기록한 revision으로 다시 받을 수 있습니다. 현재 설치된 conda 환경의 portable YAML, pip freeze와 exact build 목록을 함께 보관합니다.

현재 원본에도 `third_party/A3PRVR` 소스가 없어 백업에 포함되지 않았습니다. A3Retriever를 사용하려면 복원된 `won_woo/`에서 runtime report에 기록된 저장소와 커밋을 받으세요.

```bash
mkdir -p third_party
git clone https://github.com/chuanshen-chen/A3PRVR third_party/A3PRVR
git -C third_party/A3PRVR checkout --detach ae02a70f7d6d5320531df0113ee48e47712a072d
```

## 확인

```bash
python3 backup/test_backup_workspace.py
python3 backup/test_verify_restore.py
conda run -n holmes python backup/test_restore_links.py
```

백업 성공은 로컬 목록 생성만으로 판단하지 않습니다. GitHub remote commit 조회와 Drive 업로드·파일 검증 완료 여부를 각각 확인해야 합니다.

공식 문서: [GitHub 파일 제한](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github), [rclone copy](https://rclone.org/commands/rclone_copy/), [rclone Drive](https://rclone.org/drive/), [서버 인증 설정](https://rclone.org/remote_setup/).
