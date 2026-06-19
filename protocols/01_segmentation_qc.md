<h1 style="color: #333; text-align: center; padding: 15px 0; border-top: 3px double #333; border-bottom: 3px double #333; font-family: Georgia, serif; font-style: italic; font-weight: normal; margin-top: 20px; margin-bottom: 30px;">
  Quality Control and Optimization Protocol for Tissue Segmentation
</h1>

<div style="text-align: center; font-family: Georgia, serif; font-size: 1.2em; color: #555; margin-top: -10px; margin-bottom: 30px;">
  By <strong>Mathis BOUVET</strong> — Biologist specializing in Reproduction and Development
  <br>
  <span style="font-size: 0.8em; font-style: italic;">March 2026</span>
</div>
<br>

> **Important note**
> : This document contains no real data

<div style="border: 1px solid #569cd6; border-radius: 10px; padding: 20px; background-color: rgba(86, 156, 214, 0.1); color: #000000;">
  <strong>Cell segmentation</strong><br>
  Before initiating spatial analysis, it is crucial to obtain a segmentation that remains faithful to the biological reality of the tissue under study. While manual segmentation remains the gold standard in terms of accuracy, time constraints necessitate the use of automated tools. Solutions such as Cellpose, StarDist, or QuPath now offer high-precision segmentation powered by Deep Learning. However, these tools face certain limitations, notably the risk of overfitting or an inability to generalize across diverse tissue architectures. Segmentation requirements differ radically between a testicular section and an ovarian section, for example. Consequently, it is indispensable to develop rigorous validation protocols, where automated segmentation is evaluated against a manually established 'ground truth' to quantify its reliability.
</div>

<h2 style="color: #000000; border-bottom: 1px solid #333; font-family: Georgia, serif;  font-weight: normal; padding-bottom: 5px; margin-top: 35px;">
  Objective
</h2>
This document details the steps for cellular segmentation analysis, utilizing regions of interest (ROI) masks derived from manual segmentation to compare them against the results of automated segmentation.

<br>
<br>

<div style="border: 1px solid #d65323; border-radius: 10px; padding: 20px; background-color: rgba(213, 101, 45, 0.1); color: #000000;">
  <strong>Importing libraries</strong><br>

<details>
  <summary><b>Show/hide configuration code</b></summary>

```python
import importlib
import subprocess
import sys

# Dictionary of necessary libraries
required_packages = {
    "numpy": "numpy",
    "cv2": "opencv-python",
    "matplotlib": "matplotlib",
    "skimage": "scikit-image",
    "read_roi": "read-roi",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "scipy": "scipy",
    "seaborn": "seaborn"
}
def install_and_import():
    print("Environmental analysis nlp_env...")
    for module_name, package_name in required_packages.items():
        try:
            importlib.import_module(module_name)
            print(f"✅ {module_name} is already ready.")
        except ImportError:
            print(f"⚠️ {module_name} is missing. Installation of {package_name} in progress...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"{package_name} installed successfully !")

install_and_import()

# Importing libraries
import numpy as np
import cv2
import pandas as pd 
import random
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import os

from skimage.draw import polygon, polygon_perimeter
from read_roi import read_roi_file
from scipy.stats import ks_2samp, mannwhitneyu
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import IsolationForest

print("All modules are imported and ready to use!")
```
</details>
</div>

<h2 style="color: #000000; border-bottom: 1px solid #333; font-family: Georgia, serif;  font-weight: normal; padding-bottom: 5px; margin-top: 35px;">
  1. Creating masks from the ROIs
</h2>

### 1.a. ROIs created via Fiji and imported into the Python environment


A reference DAPI image is extracted from the MACSima system and then exported in `.tif` format for processing in Fiji. Regions of interest (ROI) are generated within it to serve as the basis for creating segmentation masks. Once these ROIs are created, both the reference image and the folder containing their coordinates are imported into the Python environment for the subsequent analysis.

> **DAPI Image**
> : Exported in `tiff` format via MACSIMA

> **SetROIs**
> : Created and exported from Fiji in `zip` format


```python
# Load the TIFF image
image = cv2.imread("[image.tiff]")
if image is None:
    raise FileNotFoundError("Image TIFF not found or unreadable")

# Temporary folder for extracting ROIs
extract_folder = "roi_dezip"

# Extract the files from the ZIP archive
with zipfile.ZipFile("[RoiSet].zip", "r") as zip_ref:
    zip_ref.extractall(extract_folder)

roi_files = [f for f in os.listdir(extract_folder) if f.endswith(".roi")]
roi_data = {}
for roi_file in roi_files:
    roi_path = os.path.join(extract_folder, roi_file)
    roi_data.update(read_roi_file(roi_path))
print(f"Number of ROIs charged : {len(roi_data)}")
```
Be sure to check the number of imported ROIs


### 1. b Generation of masks


In this analysis, several types of masks are generated to test the MACSiQView segmentation mode. The executed code yields four different types of masks.

```python
#ROIs colorful + black background
masked_image = np.zeros_like(image)
def generate_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
for roi_name, roi in roi_data.items():
    x = np.array(roi["x"])
    y = np.array(roi["y"])
    rr, cc = polygon(y, x, shape=image.shape[:2])
    color = generate_random_color()
    masked_image[rr, cc] = color
cv2.imwrite("mask_1.tif", masked_image)

#ROIs colorful (2) + black background
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
masked_image = np.zeros_like(image)
roi_color_idx = 0 
for roi_name, roi in roi_data.items():
    x = np.array(roi["x"])
    y = np.array(roi["y"])
    rr, cc = polygon(y, x, shape=image.shape[:2])
    color = colors[roi_color_idx]
    masked_image[rr, cc] = color
    roi_color_idx = (roi_color_idx + 1) % len(colors)
cv2.imwrite("mask_2.tif", masked_image)

#ROIs in grayscale
gray_mask = np.zeros(image.shape[:2], dtype=np.uint8)
for roi_name, roi in roi_data.items():
    x = np.array(roi["x"])
    y = np.array(roi["y"])
    rr, cc = polygon(y, x, shape=image.shape[:2])
    gray_mask[rr, cc] = np.clip(gray_mask[rr, cc] + 50, 0, 255)
cv2.imwrite("mask_3.tif", gray_mask)

#ROIs in grayscale + outlines
gray_mask = np.zeros(image.shape[:2], dtype=np.uint8)
for roi_name, roi in roi_data.items():
    x = np.array(roi["x"])
    y = np.array(roi["y"])
    rr, cc = polygon(y, x, shape=image.shape[:2])
    gray_mask[rr, cc] = np.clip(gray_mask[rr, cc] + 50, 0, 255)
    rr_perim, cc_perim = polygon_perimeter(y, x, shape=image.shape[:2])
    gray_mask[rr_perim, cc_perim] = 0
cv2.imwrite("mask_4.tif", gray_mask)
```
<br>

### 1.c MACSiQView segmentation on masks generated by Python and comparison between them


The MACSiQView software offers various segmentation algorithms. The objective of this study is to evaluate the four available modalities (Import mask, Import label, Single Cell, and Tissue) to identify the one providing the most faithful segmentation of biological structures. Once the segmentation is validated, the 'Feature Table' tab allows for the selection of descriptors to be included in the analysis. This step ensures the exclusive export of cellular morphological data, independent of fluorescence intensities. The resulting file, in .csv format, is then imported into a Python environment. From the set of extracted variables, the following parameters were retained: Area, Perimeter, Centroid X and Y, Feret, and Mean Intensity.

<h2 style="color: #000000; border-bottom: 1px solid #333; font-family: Georgia, serif;  font-weight: normal; padding-bottom: 5px; margin-top: 35px;">
2 Comparison of the different segmentations
</h2>

### 2.a  Calculates comparisons



First, we import the automatic segmentations `df_auto` and the manual segmentation `df_manual`. We keep only the following comparison points: Area, Perimeter, Centroid X and Y, Field and Mean intensity.

We apply normalization using `MinMaxScaler()` and construct a comparison dataframe `df_comparison`. The choice of `MinMaxScaler` is based on the assumption of prior data cleaning under MACSiQView, guaranteeing the absence of extreme outliers and allowing direct comparison on a normalized scale [0, 1] which is more readable for morphological analysis.

```python
df_manual = pd.read_csv("[Manual_segmentation].csv")

df_autos = [
    pd.read_csv("Mask_3_Single_Cell.csv"),
    pd.read_csv("Mask_3_Tissue.csv"),
    pd.read_csv("Mask_4_Import_mask.csv"),
    pd.read_csv("Mask_4_Single_Cell.csv"),
    pd.read_csv("Mask_4_Tissue.csv")
]

df_manual = df_manual.rename(columns={
    'Area': 'area',
    'Perim.': 'perimeter',
    'XM': 'centroid_x',
    'YM': 'centroid_y',
    'Feret': 'feret',
    'Mean': 'mean_intensity'
})

columns_to_keep = ['area', 'perimeter', 'centroid_x', 'centroid_y', 'feret', 'mean_intensity']
df_manual = df_manual[columns_to_keep]

print("Before normalization :")
print(df_manual.head(), "\n")

def rename_auto(df):
    return df.rename(columns={
        'Nucleus Size': 'area',
        'Nucleus Contour Length': 'perimeter',
        'Nuc X': 'centroid_x',
        'Nuc Y': 'centroid_y',
        'Nucleus Feret Diameter Max': 'feret',
        'Nucleus DNA Mean': 'mean_intensity'
    })[columns_to_keep]

df_autos = [rename_auto(df) for df in df_autos]

scaler = MinMaxScaler()
df_manual[columns_to_keep] = scaler.fit_transform(df_manual[columns_to_keep])
for df_auto in df_autos:
    df_auto[columns_to_keep] = scaler.transform(df_auto[columns_to_keep])

print("After normalization :")
print(df_manual.head())
print(df_autos[0].head())

df_comparison = df_manual[columns_to_keep].copy()
for i, df_auto in enumerate(df_autos, start=1):
    for col in columns_to_keep:
        df_comparison[f'{col}_auto{i}'] = df_auto[col]
```


### 2.b Statistical comparison of the closest segmentation (Kolmogorov-Smirnov)



We use a two-sample Kolmogorov-Smirnov test. To simplify the process, we only retrieve the KS statistic, store it in a specific dataframe `distance_df`, and add the average KS distance over the 6 parameters.

By analyzing the parameters, the script directly informs the nearest segmentation


```python
# Correspondence for new names
legend_mapping = {
    f'Auto{i+1}': name for i, name in enumerate([
        'Mask 3 - Single Cell',
        'Mask 3 - Tissue',
        'Mask 4 - Import Mask',
        'Mask 4 - Single Cell',
        'Mask 4 - Tissue'
    ])
}

# Calculates its Kolmogorov-Smirnov distances for each parameter
distances = {
    column: [ks_2samp(df_manual[column], df_auto[column])[0] for df_auto in df_autos]
    for column in columns_to_keep
}

distances_df = pd.DataFrame(distances, index=list(legend_mapping.keys()))
distances_df['Mean Distance'] = distances_df.mean(axis=1)

print("Average Kolmogorov-Smirnov distances for each segmentation :")
print(distances_df['Mean Distance'])

closest_auto = distances_df['Mean Distance'].idxmin()
print(f"\nThe segmentation closest to df_manual is : {legend_mapping[closest_auto]}")
```

### 2.c Visualization


We also perform barplots of average distances

```python
plt.figure(figsize=(10, 6))
plt.style.use('dark_background')
ax = distances_df['Mean Distance'].plot(kind='bar', color='#E9EDC9', edgecolor='white')
for p in ax.patches:
    ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points', color='white')
plt.title('Average Kolmogorov-Smirnov Distances by Segmentation', color='white')
plt.ylabel('Average Distance', color='white')
plt.xlabel('Segmentations', color='white')
plt.xticks(ticks=range(len(legend_mapping)), labels=[legend_mapping[item] for item in distances_df.index], rotation=45, ha='right', color='white')
plt.yticks(color='white')
plt.grid(True, linestyle='--', alpha=0.6, color='white')
ax.set_facecolor('black')
ax.figure.set_facecolor('black')
plt.show()
```
We also visualize density curves (KDE) to compare the shape of the distributions.
```python

df_manual_kde = pd.DataFrame(
    MinMaxScaler().fit_transform(df_manual[columns_to_keep]),
    columns=columns_to_keep
)
df_autos_kde = [
    pd.DataFrame(MinMaxScaler().fit_transform(df[columns_to_keep]), columns=columns_to_keep)
    for df in df_autos
]

# Construction of df_comparison_kde
df_comparison_kde = df_manual_kde[columns_to_keep].copy()
for i, df_auto in enumerate(df_autos_kde, start=1):
    for col in columns_to_keep:
        df_comparison_kde[f'{col}_auto{i}'] = df_auto[col]

# KDE plot
plt.style.use('default')
params = ['area', 'perimeter', 'feret', 'mean_intensity']

fig, axes = plt.subplots(2, 2, figsize=(18, 10))
fig.patch.set_facecolor('white')
axes = axes.flatten()

for idx, param in enumerate(params):
    ax = axes[idx]
    ax.set_facecolor('white')

    for i in range(1, 6):
        key = f'Auto{i}'
        colname = f'{param}_auto{i}'
        if colname in df_comparison_kde.columns:
            sns.kdeplot(df_comparison_kde[colname], label=legend_mapping[key], linewidth=2, ax=ax)
        else:
            raise KeyError(f"Missing column in df_comparison_kde : {colname}")

    if param in df_comparison_kde.columns:
        sns.kdeplot(df_comparison_kde[param], label='Manual reference', color='cyan', linestyle='--', linewidth=2, ax=ax)
    else:
        raise KeyError(f"Missing manual column : {param}")

    ax.set_title(f'Comparative Distribution - {param}')
    ax.set_xlabel('Normalized Value')
    ax.set_ylabel('Density')
    ax.grid(True, linestyle='--', alpha=0.6)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title='Segmentation', loc='upper center', ncol=3)

plt.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig('kde_comparative_distributions.png', format='png', facecolor='white')
plt.show()
```

<h2 style="color: #000000; border-bottom: 1px solid #333; font-family: Georgia, serif;  font-weight: normal; padding-bottom: 5px; margin-top: 35px;">
3. Good segmentation?
</h2>

### 3.a Comparison calculation between manual and automatic segmentation



The purpose of this study is the comparative evaluation of an automated segmentation performed in MACSiQView against a reference methodology established from manually generated Region of Interest (ROI) masks. The data from the manual segmentation, already compiled, constitute the reference dataset `df_ref`. The present analysis aims to apply an identical procedure to a DAPI image from MACSima immunofluorescence, the results of which are recorded in the test dataset df_test. By ensuring strict uniformity of export parameters, this approach allows for the quantification of automated segmentation accuracy, the validation of its scientific relevance, and the formulation of optimization hypotheses for subsequent image processing protocols.

<div style="border: 1px solid #fff200; border-radius: 10px; padding: 20px; background-color: rgba(234, 255, 0, 0.1); color: #000000;">
The verification steps integrated into the script, although they increase its complexity, ensure that the columns conform to the previously selected parameters.
</div>

<br>

Once the columns are sorted and the data normalized (this is mandatory), we use Isolation Forest, which will be trained only on the `X_ref_scaled` reference (since we manually segmented only one image). We assume a 10% contamination rate.

To verify sensitivity to the contamination parameter, the isolation model is run on different contamination values. For each value, the percentage of test cells classified as "OK" is checked.

We import the reference dataset df_ref and the test dataset df_test. We verify that all 14 morphological features are present in both files. The data are then normalized using StandardScaler fit is applied only to X_ref, while transform is applied to X_test, ensuring that the reference scale remains that of the manual segmentation.

```python 
import warnings
warnings.filterwarnings("ignore")

df_ref = pd.read_csv("Segmentation(1).csv")
df_test = pd.read_csv("Segmentation_EH3524.csv")

df_ref.columns = df_ref.columns.str.strip()
df_test.columns = df_test.columns.str.strip()

features_to_use = [
    "Cell Bbox X Size", "Cell Bbox Y Size", "Cell Shape Circle Like", "Cell Shape Ellipse Like",
    "Cell Shape Elongation", "Cell Shape Square Like", "Cell Shape Triangle Like",
    "Cell Size", "Nucleus Size", "Nucleus Roundness", "Nucleus Convexity", "Cell Convexity",
    "Quality Cell In-Focus", "Quality Nuclear Segmentation"
]

missing = [col for col in features_to_use if col not in df_ref.columns or col not in df_test.columns]
if missing:
    raise ValueError(f"Missing columns in files : {missing}")

X_ref = df_ref[features_to_use].dropna()
X_test = df_test[features_to_use].dropna()
df_test_clean = df_test.loc[X_test.index].copy()

scaler = StandardScaler()
X_ref_scaled = scaler.fit_transform(X_ref)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Reference : {len(X_ref)} cells · Test : {len(X_test)} cells")
```
Before setting the contamination parameter, we assess the model's sensitivity by running the Isolation Forest across a range of contamination values (0.01 → 0.20). For each value, we examine the percentage of test cells classified as "OK." This approach helps identify a contamination level that is biologically meaningful and consistent with the data.

```python
# Sensitivity analysis choosing best contamination
contamination_values = [0.01, 0.05, 0.10, 0.15, 0.20]
ok_percentages = []

for c in contamination_values:
    model = IsolationForest(contamination=c, random_state=42)
    model.fit(X_ref_scaled)
    preds = model.predict(X_test_scaled)
    ok_percent = (preds == 1).sum() / len(preds) * 100
    ok_percentages.append(ok_percent)

plt.figure(figsize=(8, 4))
plt.plot(contamination_values, ok_percentages, marker='o', color='steelblue')
plt.xlabel("Contamination rate")
plt.ylabel("% well-segmented cells")
plt.title("Contamination sensitivity analysis")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('contamination_sensitivity.png', format='png')
plt.show()

print("Contamination · % OK")
for c, pct in zip(contamination_values, ok_percentages):
    print(f"  {c:.2f} → {pct:.1f}%")
```
Once the contamination value has been selected, the final model is trained exclusively on X_ref_scaled. In addition to the -1/+1 labels, we retrieve a continuous anomaly score using decision_function — the more negative the score, the more abnormal the cell is considered. The results are then exported to segmentation_test_annotated.csv.

```python
# Final model with chosen contamination
best_contamination = 0.10

model = IsolationForest(contamination=best_contamination, random_state=42)
model.fit(X_ref_scaled)
preds = model.predict(X_test_scaled)
scores = model.decision_function(X_test_scaled)

df_test_clean["Segmentation_OK"] = preds
df_test_clean["Anomaly_Score"] = scores

nb_total = len(preds)
nb_valides = (preds == 1).sum()
pourcentage = nb_valides / nb_total * 100

print(f"\n✅ Result with contamination={best_contamination:.2f} :")
print(f"{nb_valides} / {nb_total} cells considered well-segmented ({pourcentage:.2f}%)")

# Anomaly score distribution
plt.figure(figsize=(10, 4))
plt.hist(scores, bins=50, color='steelblue', edgecolor='white')
plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Decision threshold')
plt.xlabel('Anomaly Score')
plt.ylabel('Count')
plt.title('Distribution of Isolation Forest Anomaly Scores')
plt.legend()
plt.tight_layout()
plt.savefig('anomaly_scores_distribution.png', format='png')
plt.show()

df_test_clean.to_csv("segmentation_test_annotated.csv", index=False)
print("💾 Result saved in : segmentation_test_annotated.csv")
```


### 3.b Parameter improvement suggestion



If the Isolation Forest model identifies parameter vectors as outliers or non-compliant with optimal segmentation, it then becomes possible to extract optimization indicators for the segmentation settings. To this end, a mapping dictionary is established to link each measured variable to a specific MACSiQView parameter, thereby determining the required adjustment (incrementing or decrementing the parameter). The Mann-Whitney U statistical test is employed to compare the distributions of compliant (OK) and non-compliant (KO) populations. If a statistically significant difference is observed ($p < 0.01$), analyzing the position of the medians (or means) allows for defining the polarity of the deviation ($KO < OK$ or $KO > OK$). This divergence is then translated into operational recommendations for MACSiQView, accompanied by the calculation of the relative percentage variation between the two groups.

```python
param_map = {
    "Nucleus Size": ("Diamètre min / max", "↑ diamètre min", "↑ diamètre max"),
    "Cell Size": ("Diamètre min / max", "↑ diamètre min", "↑ diamètre max"),
    "Cell Bbox X Size": ("Diamètre max", "↑ diamètre min", "↑ diamètre max"),
    "Cell Bbox Y Size": ("Diamètre max", "↑ diamètre min", "↑ diamètre max"),
    "Nucleus Roundness": ("Séparation / Smoothing", "↑ séparation", "↓ séparation ou ↓ smoothing"),
    "Nucleus Convexity": ("Smoothing filter sigma", "↑ sigma", "↓ sigma"),
    "Cell Convexity": ("Smoothing filter sigma", "↑ sigma", "↓ sigma"),
    "Cell Shape Ellipse Like": ("Contours / Smoothing", "↑ smoothing", "↓ smoothing"),
    "Cell Shape Circle Like": ("Contours / Smoothing", "↑ smoothing", "↓ smoothing"),
    "Cell Shape Elongation": ("Contours / Séparation", "↓ séparation", "↑ séparation"),
    "Cell Shape Square Like": ("Contours", "-", "-"),
    "Cell Shape Triangle Like": ("Contours", "-", "-"),
    "Quality Cell In-Focus": ("Qualité image (acquisition)", "-", "-"),
    "Quality Nuclear Segmentation": ("Sensibilité / Smoothing", "↑ sensibilité", "↓ sensibilité"),
}
df = df_test_clean.copy()
df = df.dropna(subset=features_to_use + ['Segmentation_OK'])
df['Segmentation_Label'] = df['Segmentation_OK'].map({1: 'OK', -1: 'KO'})
summary = []
for feature in features_to_use:
    ok_vals = df[df['Segmentation_Label'] == 'OK'][feature]
    ko_vals = df[df['Segmentation_Label'] == 'KO'][feature]
    if len(ok_vals) < 10 or len(ko_vals) < 10:
        continue
    stat, p = mannwhitneyu(ok_vals, ko_vals, alternative='two-sided')
    mean_ok = ok_vals.mean()
    mean_ko = ko_vals.mean()
    direction = "-"
    suggestion = "-"
    percent_change = "-"
    param = param_map.get(feature, ("-", "-", "-"))[0]

    if p < 0.01:
        if mean_ko < mean_ok:
            direction = "KO < OK"
            suggestion = param_map[feature][1]
            percent_change = f"+{round((1 - mean_ko / mean_ok) * 100, 1)} %"
        elif mean_ko > mean_ok:
            direction = "KO > OK"
            suggestion = param_map[feature][2]
            percent_change = f"+{round((mean_ko / mean_ok - 1) * 100, 1)} %"

    summary.append({
        "Variable": feature,
        "Mean OK": round(mean_ok, 2),
        "Mean KO": round(mean_ko, 2),
        "Difference": direction,
        "MACSiQ parameter linked": param,
        "Suggested adjustment": suggestion,
        "indicative % change": percent_change,
        "p-value": round(p, 4)
    })
summary_df = pd.DataFrame(summary).sort_values("p-value")

# Console display
print("\n Summary of MACSiQ parameters to adjust :\n")
print(summary_df.to_string(index=False))

# Export CSV
summary_df.to_csv("macsiq_param_suggestions.csv", index=False)
print("\n💾 Summary exported to : macsiq_param_suggestions.csv")
```

<h2 style="color: #000000; border-bottom: 1px solid #333; font-family: Georgia, serif;  font-weight: normal; padding-bottom: 5px; margin-top: 35px;">
4. Possible improvement and limit
</h2>



### 4.a Limitations on the comparison model

While Kolmogorov-Smirnov tests and Kernel Density Estimation (KDE) visualizations allow for a global validation of segmentation fidelity, these approaches have an intrinsic limitation: they treat parameters in isolation (univariate analysis). In reality, an 'aberrant' cell is not necessarily detectable based on a single criterion (e.g., a normal surface area), but rather through an inconsistent combination of several factors (e.g., a normal surface area associated with extremely low circularity and heterogeneous intensity). The human eye spots these anomalies instinctively, but classical statistical analysis can overlook these artifacts, thereby biasing the final results of the spatial analysis.

To overcome this limitation, the Isolation Forest algorithm could be utilized. Unlike methods that seek to define a 'perfect cell' model, this Machine Learning algorithm identifies anomalies through their isolation. In a mathematical space where each morphological parameter represents a dimension, the algorithm randomly partitions the data: normal cells, which are dense and similar, require many steps to be isolated, whereas segmentation errors (artifacts, doublets, debris) are isolated much more quickly.

<div style="border: 1px solid #fff200; border-radius: 10px; padding: 20px; background-color: rgba(234, 255, 0, 0.1); color: #000000;">
We utilize the sklearn.ensemble.IsolationForest function, which is already partially in use, to address the univariate limitation. In this case, it would be used as a pre-filtering step. Then, model.predict(X_test_scaled) yields labels of -1 or 1. This labeling will then be incorporated into the KDE plots and KS tests.
</div>

### 4.b Evolution of the model towards Deep Learning

The final step of this protocol consists of moving beyond single-image validation to build a robust reference model. Currently, our analysis relies on a manually segmented image; however, tissue diversity necessitates a broader database. By compiling manual segmentations from multiple tissue sections, we can build a massive training dataset. This dataset will be used to train a Deep Learning model specifically tailored to our acquisition conditions. The workflow thus becomes cyclic: each new image is validated by the Isolation Forest and corrected by a human expert. The more varied cellular structures the model encounters, the greater its generalization capacity becomes, thereby reducing the need for manual correction. Ultimately, this self-learning model standardizes segmentation quality across all unit projects, guaranteeing total reproducibility, regardless of the operator or the biological variability of the tissue.

<div style="border: 1px solid #fff200; border-radius: 10px; padding: 20px; background-color: rgba(234, 255, 0, 0.1); color: #000000;">
We no longer use CSV files but rather models trained on pixels. Functions<strong>cellpose.models.CellposeModel</strong> or <strong>stardist.models.StarDist2D</strong> can be used. This requires ROIs transformed into binary masks
</div>
<br>
To go further, it will be necessary to incorporate tissue morphology into our model. In this way, we can weight our analysis by forcing the algorithm to assign a higher statistical weight to reference images originating from the same organ. With each new analysis of an unknown tissue, the manually validated segmentations are labeled and integrated into the library. The system thus enriches itself organically, filling its own gaps as research projects progress.

<br>
<div style="border: 1px solid #fff200; border-radius: 10px; padding: 20px; background-color: rgba(234, 255, 0, 0.1); color: #000000;">
Once the DeepLearning model is built, you simply need to use model dictionaries via functions like<strong>joblib.dump</strong>
</div>
