# Model, Dataset, and Biometric Privacy Notes

## 1. InsightFace Pretrained Weights Licensing

The InsightFace Python library code is distributed under the MIT License. However, the pretrained neural network weights provided by InsightFace (specifically the `buffalo_l` model pack containing `det_10g.onnx` and `w600k_r50.onnx`) are **subject to separate licensing restrictions**:

- **License Scope:** The pretrained weights are released strictly for **non-commercial academic research and educational use**.
- **Commercial Use:** Commercial deployment of InsightFace pretrained weights requires an official commercial license directly from InsightFace / DeepInsight.
- **Reference:**
  - Official Repository: https://github.com/deepinsight/insightface
  - Model Zoo Terms: https://github.com/deepinsight/insightface/blob/master/python-package/docs/model_zoo.md
  - Commercial Licensing: https://www.insightface.ai/solutions/face-recognition-licensing

## 2. Kaggle Dataset Licensing (`vasukipatel/face-recognition-dataset`)

- **Origin:** The dataset is a collection of public celebrity photographs scraped from the web, containing 31 public figures across 2,562 images.
- **Usage Restrictions:** The images are intended strictly for benchmarking, educational experimentation, and machine learning research.
- **Likeness and Publicity Rights:** Celebrity images remain subject to the publicity and privacy rights of the depicted individuals and the copyright of the original photographers. They are not cleared for commercial redistribution or commercial recognition products.

## 3. Biometric Data Privacy and Responsible Handling

Face images and 512-dimensional face embeddings constitute **sensitive biometric data**:
- **Local Storage Only:** All face images, embeddings (`data/memory.json`, `data/validation.json`, `data/evaluation.json`), and model artifacts (`models/classifier_v*.joblib`) are stored locally on disk and are never uploaded to cloud endpoints or public repositories.
- **Consent:** Any user-registered student photos should be collected only with explicit informed consent.
- **Protection:** Keep the `data/` and `models/` directories private. Do not commit biometric embeddings or photos to public version control.
- **Decision Integrity:** Face recognition predictions must be treated as probabilistic suggestions and should never be used for automated punitive decisions or unverified attendance logging.

## 4. Labeled Faces in the Wild (LFW) Licensing (`jessicali9530/lfw-dataset`)

- **Origin & Authors:** Gary B. Huang, Manu Ramesh, Tamara Berg, and Erik Learned-Miller. *Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments*. University of Massachusetts, Amherst, Technical Report 07-49, 2007.
- **Reference:** http://vis-www.cs.umass.edu/lfw/
- **Licensing & Permitted Use:** The LFW dataset is provided freely for **non-commercial academic and research purposes**.
- **Copyright & Likeness:** The images in the database are collected from Yahoo! News and are copyrighted by the respective photographers/news organizations. They may not be used for commercial biometric products or commercial advertising.
