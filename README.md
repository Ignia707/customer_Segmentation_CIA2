# Customer Segmentation using Text-Based Transaction Analysis

A data mining project focused on customer segmentation using unstructured billing data and bio-texts. The aim is to enhance targeted marketing by clustering customers based on transaction patterns and bios, using a combination of classical text mining, feature engineering, and machine learning techniques.

---

## Problem Statement

Retail businesses generate large volumes of raw transaction records (like bills) and collect customer information (bios). These unstructured and noisy datasets pose challenges for extracting valuable insights.

**Goal**: Cluster customers based on their transaction behavior and bio-texts to enable intelligent, targeted marketing.

---

## Challenges faced

- **Unstructured Text Files**: Raw bills required parsing using regex.
- **Data Inconsistencies**: Missing values, duplicate entries, and inconsistent formats were cleaned.
- **Mixed Data Types**: Numerical (RFM), categorical (gender, region), and textual (bios).
- **Noisy Text in Bios**: Slang, emojis, and casing normalized using custom functions.
- **High Dimensionality**: TF-IDF and encoding caused large feature spaces — resolved with PCA.
- **Cluster Evaluation**: Used Silhouette Score and visual inspection with PCA-based 3D visualization.

---

## Architecture Overview

```plaintext
1. Parse raw bills using regex → Transaction DataFrame
2. Join with customer data on ID
3. Process:
   a. Transaction data: RFM + Association Rule Mining
   b. Bios: Text normalization + TF-IDF
4. Combine features
5. Apply PCA
6. Cluster using KMeans, DBSCAN, Agglomerative, GMM
7. Evaluate with Silhouette Score and visualize
