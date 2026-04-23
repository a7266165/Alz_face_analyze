# Dataset Download Status (2026-04-07)

## Successfully Downloaded

| Dataset | Size | Contents | Notes |
|---------|------|----------|-------|
| AFAD | 1.5GB (compressed) | 45 split tar.xz files, need `cat AFAD-Full.tar.xz* > AFAD-Full.tar.xz && tar xf AFAD-Full.tar.xz` | 164K+ Asian faces, ages 15-40 |
| AAF | 8.4MB | README + illustration (images need separate download from authors) | 13K faces, ages 2-80 |
| AFD | 11MB | Paper PDF + sample (images need separate download from authors) | Large-scale Asian faces |
| FairFace | 556MB | fairface_padding025.zip + train/val labels CSV | 108K faces, has East/SE Asian + age labels, CC BY 4.0 |
| UTKFace | 1.3GB | 3 tar.gz parts (in-the-wild) | 20K+ faces, filename format: age_gender_race_date.jpg (race=2 is Asian) |
| IMDB-WIKI | ~8GB | wiki_crop.tar (773MB) + imdb_crop.tar (downloading, 7GB) | 523K faces with age/gender labels |
| IMDB-Clean | 245MB | Clean label CSVs + scripts | Cleaned version of IMDB-WIKI labels |
| SZU-EmoDage | 585MB | Aging_faces.zip (35M) + Emotional_faces.zip (24M) + Dynamic_faces.zip (526M) | Chinese faces with age 60-70 variants, CC BY 4.0 |
| LAOFIW | 1.5MB | Metadata only (age + gender CSV) | Images no longer available from VGG |
| MegaAge-Asian | 277MB | megaage_asian.zip | 40K Asian faces with age labels |
| FaceAge | 15MB | Model code (Python) | Biological age estimation from faces |
| Face-Age-10K | 394MB | Parquet file with embedded images | ~9K faces with age group labels (56-65, 66-80, 80+) |
| GIRAF | 66MB | GIRAF_faces.zip (186 images, 3 age groups) | AI-generated same-person aging faces |

## Failed to Download (Need Manual Download)

| Dataset | Reason | How to Get |
|---------|--------|-----------|
| CACD | Google Drive permission denied (too many downloads) | Download manually: https://drive.google.com/file/d/1hYIZadxcPG27Fo7mQln0Ey7uqw1DoBvM |
| UTKFace (aligned/cropped) | Google Drive permission denied | Use Kaggle mirror: kaggle datasets download -d jangedoo/utkface-new |

## Requires Application/Contact

| Dataset | How to Get |
|---------|-----------|
| Tsinghua-FED | Email authors (see PLOS ONE paper) |
| Diverse Asian Facial Ages | Kaggle login required |
| AgeDB | Apply to authors |
| MORPH II | Apply to UNC |
| FG-NET | Apply to authors |
| APPA-REAL | ChaLearn LAP website |
| Adience | Flickr-sourced, academic use |
| ETRI-Activity3D | Apply to ETRI |
| NTU RGB+D | Register at rose1.ntu.edu.sg |
| K-FACE | Register at kface.kist.re.kr |
| ADReFV | Contact paper authors |
| PROMPT Dataset | Contact Keio Medical School |
| Tokyo U AD Dataset | Contact U of Tokyo |
| YT-DemTalk | Contact paper authors |
| DementiaBank | Register at talkbank.org |
| ElderReact | Fill request form on GitHub |
| RFW | Contact dataset creators |
| CALFW | Download from whdeng.cn |
| Park Aging Mind | Request from UT Dallas |

## Post-Download Steps

1. **AFAD**: Reassemble and extract: `cd AFAD && cat AFAD-Full.tar.xz* > AFAD-Full.tar.xz && tar xf AFAD-Full.tar.xz`
2. **FairFace**: Unzip and filter by race (East Asian/SE Asian) + age (60+)
3. **UTKFace**: Extract tar.gz parts, filter by filename (race=2 for Asian, age>=60)
4. **IMDB-WIKI**: Extract tar files, use IMDB-Clean labels for filtering
5. **MegaAge-Asian**: Unzip
6. **SZU-EmoDage**: Unzip all 3 files
7. **GIRAF**: Unzip
