#!/usr/bin/env python
# coding: utf-8

# # Target Ensemble
# 
# Apart from the main target, there are actually many auxilliary targets in the dataset.
# 
# These targets are fundamentally related to the main target which make them potentially helpful to model. And because these targets have a wide range of correlations to the main targets, it means that we could potentially build some nice ensembles to boost our performance.
# 
# In this notebook, we will
# 1. Explore the auxilliary targets
# 2. Select our favorite targets to include in the ensemble
# 3. Create an ensemble of models trained on different targets
# 4. Pickle and upload our ensemble model

#%%


# Install dependencies
get_ipython().system('pip install -q numerapi pandas pyarrow matplotlib lightgbm scikit-learn cloudpickle==2.2.1 seaborn scipy==1.10.1')

# Inline plots
get_ipython().run_line_magic('matplotlib', 'inline')


# ## 1. Auxilliary Targets
# 
# Let's start by taking a look at the different targets in the training data.

#%%


import pandas as pd
import json
from numerapi import NumerAPI

# Set the data version to one of the most recent versions
DATA_VERSION = "v5.0"
MAIN_TARGET = "target_cyrusd_20"
TARGET_CANDIDATES = [
  MAIN_TARGET,
  "target_victor_20",
  "target_xerxes_20",
  "target_teager2b_20"
]
FAVORITE_MODEL = "v5_lgbm_ct_blend"

# Download data
napi = NumerAPI()
napi.download_dataset(f"{DATA_VERSION}/train.parquet")
napi.download_dataset(f"{DATA_VERSION}/features.json")

# Load data
feature_metadata = json.load(open(f"{DATA_VERSION}/features.json"))
feature_cols = feature_metadata["feature_sets"]["small"]
# use "medium" or "all" for better performance. Requires more RAM.
# features = feature_metadata["feature_sets"]["medium"]
# features = feature_metadata["feature_sets"]["all"]
target_cols = feature_metadata["targets"]
train = pd.read_parquet(
    f"{DATA_VERSION}/train.parquet",
    columns=["era"] + feature_cols + target_cols
)

# Downsample to every 4th era to reduce memory usage and speedup model training (suggested for Colab free tier)
# Comment out the line below to use all the data (higher memory usage, slower model training, potentially better performance)
train = train[train["era"].isin(train["era"].unique()[::4])]

# Print target columns
train[["era"] + target_cols]


# ### The main target

# First thing to note is that `target` is just an alias for the `cyrus` target, so we can drop this column for the rest of the notebook.

#%%


# Drop `target` column
assert train["target"].equals(train[MAIN_TARGET])
targets_df = train[["era"] + target_cols]


# ### Target names
# 
# At a high level, each target represents a different kind of stock market return
# - the `name` represents the type of stock market return (eg. residual to market/country/sector vs market/country/style)
# - the `_20` or `_60` suffix denotes the time horizon of the target (ie. 20 vs 60 market days)
# 
# The reason why `cyrus` as our main target is because it most closely matches the type of returns we want for our hedge fund. Just like how we are always in search for better features to include in the dataset, we are also always in search for better targets to make our main target. During our research, we often come up with targets we like but not as much as the main target, and these are instead released as auxilliary targets.

#%%


# Print target names grouped by name and time horizon
pd.set_option('display.max_rows', 100)
t20s = [t for t in target_cols if t.endswith("_20")]
t60s = [t for t in target_cols if t.endswith("_60")]
names = [t.replace("target_", "").replace("_20", "") for t in t20s]
pd.DataFrame({"name": names,"20": t20s,"60": t60s}).set_index("name")


# ### Target values
# 
# Note that some targets are binned into 5 bins while others are binned into 7 bins.
# 
# Unlike feature values which are integers ranging from 0-4, target values are floats which range from 0-1.

#%%


# Plot target distributions
targets_df[TARGET_CANDIDATES].plot(
  title="Target Distributions",
  kind="hist",
  bins=35,
  density=True,
  figsize=(8, 4),
  subplots=True,
  layout=(2, 2),
  ylabel="",
  yticks=[]
)


# It is also important to note that the auxilary targets can be `NaN`, but the primary target will never be `NaN`. Since we are using tree-based models here we won't need to do any special pre-processing.

#%%


# print number of NaNs per era
nans_per_era = targets_df.groupby("era").apply(lambda x: x.isna().sum())
nans_per_era[target_cols].plot(figsize=(8, 4), title="Number of NaNs per Era", legend=False)


# ### Target correlations
# 
# The targets have a wide range of correlations with each other even though they are all fundamentally related, which should allow the construction of diverse models that ensemble together nicely.

#%%


# Plot correlation matrix of targets
import seaborn as sns
sns.heatmap(
  targets_df[target_cols].corr(),
  cmap="coolwarm",
  xticklabels=False,
  yticklabels=False
)


# Since we are ultimately trying to predict the main target, it is perhaps most important to consider each auxilliary target's correlation to it.

#%%


(
    targets_df[target_cols]
    .corrwith(targets_df[MAIN_TARGET])
    .sort_values(ascending=False)
    .to_frame("corr_with_cyrus_v4_20")
)


# ## 2. Target Selection
# 
# Our goal is to create an ensemble of models trained on different targets. But which targets should we use?
# 
# When deciding which model to ensemble, we should consider a few things:
# 
# - The performance of the predictions of the model trained on the target vs the main target
# 
# - The correlation between the target and the main target
# 
# To keep things simple and fast, let's just arbitrarily pick a few 20-day targets to evaluate.

# ### Model training and generating validation predictions
# 
# Like usual we train on the training dataset, but this time we do it for each target.
# 
# Sit back and relax, this will take a while
# # ☕

#%%


import lightgbm as lgb

models = {}
for target in TARGET_CANDIDATES:
    model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=5,
        num_leaves=2**4-1,
        colsample_bytree=0.1
    )
    # We've found the following "deep" parameters perform much better, but they require much more CPU and RAM
    # model = lgb.LGBMRegressor(
    #     n_estimators=30_000,
    #     learning_rate=0.001,
    #     max_depth=10,
    #     num_leaves=2**10,
    #     colsample_bytree=0.1
    #     min_data_in_leaf=10000,
    # )
    model.fit(
        train[feature_cols],
        train[target]
    )
    models[target] = model


# Then we will generate predictions on the validation dataset.

#%%


# Download validation data
napi.download_dataset(f"{DATA_VERSION}/validation.parquet")

# Load the validation data, filtering for data_type == "validation"
validation = pd.read_parquet(
    f"{DATA_VERSION}/validation.parquet",
    columns=["era", "data_type"] + feature_cols + target_cols
)
validation = validation[validation["data_type"] == "validation"]
del validation["data_type"]

# Downsample every 4th era to reduce memory usage and speedup validation (suggested for Colab free tier)
# Comment out the line below to use all the data
validation = validation[validation["era"].isin(validation["era"].unique()[::4])]

# Embargo overlapping eras from training data
last_train_era = int(train["era"].unique()[-1])
eras_to_embargo = [str(era).zfill(4) for era in [last_train_era + i for i in range(4)]]
validation = validation[~validation["era"].isin(eras_to_embargo)]

# Generate validation predictions for each model
for target in TARGET_CANDIDATES:
    validation[f"prediction_{target}"] = models[target].predict(validation[feature_cols])

pred_cols = [f"prediction_{target}" for target in TARGET_CANDIDATES]
validation[pred_cols]


# ### Evaluating the performance of each model
# 
# Now we can evaluate the performance of our models.

#%%


# install Numerai's open-source scoring tools
get_ipython().system('pip install -q --no-deps numerai-tools')

# import the 2 scoring functions
from numerai_tools.scoring import numerai_corr, correlation_contribution


# As you can see in the performance chart below, models trained on the auxiliary target are able to predict the main target pretty well, but the model trained on the main target performs the best.

#%%


prediction_cols = [
    f"prediction_{target}"
    for target in TARGET_CANDIDATES
]
correlations = validation.groupby("era").apply(
    lambda d: numerai_corr(d[prediction_cols], d["target"])
)
cumsum_corrs = correlations.cumsum()
cumsum_corrs.plot(
  title="Cumulative Correlation of validation Predictions",
  figsize=(10, 6),
  xticks=[]
)


# Looking at the summary metrics below:
# - the models trained on `victor` and `xerxes` have the highest means, but `victor` is less correlated with `cyrus` than `xerxes` is, which means `victor` could be better in ensembling
# - the model trained on `teager` has the lowest mean, but `teager` is significantly less correlated with `cyrus` than any other target shown

#%%


def get_summary_metrics(scores, cumsum_scores):
    summary_metrics = {}
    # per era correlation between predictions of the model trained on this target and cyrus
    mean = scores.mean()
    std = scores.std()
    sharpe = mean / std
    rolling_max = cumsum_scores.expanding(min_periods=1).max()
    max_drawdown = (rolling_max - cumsum_scores).max()
    return {
        "mean": mean,
        "std": std,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }

target_summary_metrics = {}
for pred_col in prediction_cols:
  target_summary_metrics[pred_col] = get_summary_metrics(
      correlations[pred_col], cumsum_corrs[pred_col]
  )
  # per era correlation between this target and cyrus
  mean_corr_with_cryus = validation.groupby("era").apply(
      lambda d: d[pred_col].corr(d[MAIN_TARGET])
  ).mean()
  target_summary_metrics[pred_col].update({
      "mean_corr_with_cryus": mean_corr_with_cryus
  })


pd.set_option('display.float_format', lambda x: '%f' % x)
summary = pd.DataFrame(target_summary_metrics).T
summary


# ### Selecting our favorite target
# Based on our observations above, it seems like target `victor` is the best candidate target for our ensemble since it has great performance and it is not too correlated with `cyrus`. However, it's interesting to look at how models that are very uncorrelated ensemble together we are going to also look at how `teager` ensembles with `cyrus`.
# 
# What do you think?
# 
# Note that this target selection heuristic is extremely basic. In your own research, you will most likely want to consider all targets instead of just our favorites, and may want to experiment with different ways of selecting your ensemble targets.

# ## 3. Ensembling
# 
# Now that we have reviewed and selected our favorite targets, let's ensemble our predictions and re-evaluate performance.

# ### Creating the ensemble
# 
# For simplicity, we will equal weight the predictions from target `victor` and `cyrus`. Note that this is an extremely basic and arbitrary way of selecting ensemble weights. In your research, you may want to experiment with different ways of setting ensemble weights.
# 
# Tip: remember to always normalize (percentile rank) your predictions before averaging so that they are comparable!

#%%


# Ensemble predictions together with a simple average
validation["ensemble_cyrus_victor"] = (
    validation
    .groupby("era")[[
      f"prediction_{MAIN_TARGET}",
      "prediction_target_victor_20",
    ]]
    .rank(pct=True)
    .mean(axis=1)
)
validation["ensemble_cyrus_teager"] = (
    validation
    .groupby("era")[[
      f"prediction_{MAIN_TARGET}",
      "prediction_target_teager2b_20",
    ]]
    .rank(pct=True)
    .mean(axis=1)
)

# Print the ensemble predictions
prediction_cols = [
  "prediction_target_cyrusd_20",
  "prediction_target_victor_20",
  "prediction_target_teager2b_20",
  "ensemble_cyrus_victor",
  "ensemble_cyrus_teager"
]
validation[prediction_cols]


# ### Evaluating performance of the ensemble
# Looking at the performance chart below, we can see that the peformance of our ensembles are better than that of the models trained on individual targets. Is this a result you would have expected or does it surprise you?

#%%


correlations = validation.groupby("era").apply(
    lambda d: numerai_corr(d[prediction_cols], d["target"])
)
cumsum_corrs = correlations.cumsum()

cumsum_corrs = pd.DataFrame(cumsum_corrs)
cumsum_corrs.plot(
    title="Cumulative Correlation of validation Predictions",
    figsize=(10, 6),
    xticks=[]
)


# Looking at the summary metrics below, we can see that our ensemble seems to have better `mean`, `sharpe`, and `max_drawdown` than our original model. Much more interestingly, however, is that our ensemble with `teager` has even higher sharpe than the ensemble with `victor`!

#%%


summary_metrics = get_summary_metrics(correlations, cumsum_corrs)
pd.set_option('display.float_format', lambda x: '%f' % x)
summary = pd.DataFrame(summary_metrics)
summary


# You can see below that ensembling also improves MMC performance significantly.

#%%


from numerai_tools.scoring import correlation_contribution

# Download and join in the meta_model for the validation eras
napi.download_dataset(f"v4.3/meta_model.parquet", round_num=842)
validation["meta_model"] = pd.read_parquet(
    f"v4.3/meta_model.parquet"
)["numerai_meta_model"]

def get_mmc(validation, meta_model_col):
    # Compute the per-era mmc between our predictions, the meta model, and the target values
    per_era_mmc = validation.dropna().groupby("era").apply(
        lambda x: correlation_contribution(
            x[prediction_cols], x[meta_model_col], x["target"]
        )
    )

    cumsum_mmc = per_era_mmc.cumsum()

    # compute summary metrics
    summary_metrics = get_summary_metrics(per_era_mmc, cumsum_mmc)
    summary = pd.DataFrame(summary_metrics)

    return per_era_mmc, cumsum_mmc, summary

per_era_mmc, cumsum_mmc, summary = get_mmc(validation, "meta_model")
# plot the cumsum mmc performance
cumsum_mmc.plot(
  title="Cumulative MMC of Neutralized Predictions",
  figsize=(10, 6),
  xticks=[]
)

pd.set_option('display.float_format', lambda x: '%f' % x)
summary


# #### Benchmark Models
# 
# It's no accident that a model trained on `teager` nicely ensembles with `cyrus`. We have seen in our research that models trained or ensembled using `teager` perform well. We even released a benchmark for a [teager ensemble](https://numer.ai/v42_teager_ensemble). We submit predictions for all internally known models [here](https://numer.ai/~benchmark_models) and release files with their predictions.

#%%


# download Numerai's benchmark models
napi.download_dataset(f"{DATA_VERSION}/validation_benchmark_models.parquet")
benchmark_models = pd.read_parquet(
    f"{DATA_VERSION}/validation_benchmark_models.parquet"
)
benchmark_models


# Because models trained on newer targets perform so well and we release their predictions, it's likely many users will begin to shift their models to include newer data and targets. By extension, the Meta Model will begin to include information from from these new targets.
# 
# This means that MMC over the validation period may not be truly indicative of out-of-sample performance. The Meta Model over the early validation period did not have access to newer data/targets and MMC over the validation period may be misleading.
# 
# So if the Meta Model was much closer to our teager ensemble, what would your MMC look like?

#%%


validation[FAVORITE_MODEL] = benchmark_models[FAVORITE_MODEL]


per_era_mmc, cumsum_mmc, summary = get_mmc(validation, FAVORITE_MODEL)
# plot the cumsum mmc performance
cumsum_mmc.plot(
  title="Contribution of Neutralized Predictions to Numerai's Teager Ensemble",
  figsize=(10, 6),
  xticks=[]
)

pd.set_option('display.float_format', lambda x: '%f' % x)
summary


# Ouch. Our teager models actually perform the worst. This means we aren't adding very useful signal to a model that Numerai already created, but this should not be surprising since we are training basically the same model. The model trained with `xerxes`, however, still does well against Numerai's model. What do you think this means?
# 
# It's also helpful to if we measured the contribution of your models to all of Numerai's benchmark models. We call this Benchmark Model Contribution or `BMC`.  On the website, `BMC` measures your model's contribution to a weighted ensemble of all of our Benchmark Models.
# 
# This is an important metric to track because it tells you how additive your model is to Numerai's known models and, by extension, how additive you might be to the Meta Model in the future.
# 
# To keep things simple, we will use an unweighted ensemble of Numerai's Benchmarks to measure your models' BMC, let's take a look:

#%%


validation["numerai_benchmark"] = (
    benchmark_models
    .groupby("era")
    .apply(lambda x: x.mean(axis=1))
    .reset_index()
    .set_index("id")[0]
)

per_era_mmc, cumsum_mmc, summary = get_mmc(validation, "numerai_benchmark")
# plot the cumsum mmc performance
cumsum_mmc.plot(
  title="Cumulative BMC of Neutralized Predictions",
  figsize=(10, 6),
  xticks=[]
)

pd.set_option('display.float_format', lambda x: '%f' % x)
summary


# Looking at the results above, none of these models seem very additive to models that Numerai can already create. It will take some research and experimentation to find something additive to Numerai's benchmarks.
# 
# Ensembling models trained on different targets can be a very fruitful avenue of research. However, it is completely up to you whether or not to create an ensemble - there are many great performing models that don't make use of the auxilliary targets at all.
# 
# If you are interested in learning more about targets, we highly encourage you to read up on these forum posts
# - https://forum.numer.ai/t/how-to-ensemble-models/4034
# - https://forum.numer.ai/t/target-jerome-is-dominating-and-thats-weird/6513

# ## 4. Model Upload
# To wrap up this notebook, let's pickle and upload our ensemble.
# 
# As usual, we will be wrapping our submission pipeline into a function. Since we already have our favorite targets and trained models in memory, we can simply reference them in our function.  

#%%


# we now give you access to the live_benchmark_models if you want to use them in your ensemble
def predict_ensemble(
    live_features: pd.DataFrame,
    live_benchmark_models: pd.DataFrame
) -> pd.DataFrame:
    favorite_targets = [
        'target_cyrusd_20',
        'target_teager2b_20'
    ]
    # generate predictions from each model
    predictions = pd.DataFrame(index=live_features.index)
    for target in favorite_targets:
        predictions[target] = models[target].predict(live_features[feature_cols])
    # ensemble predictions
    ensemble = predictions.rank(pct=True).mean(axis=1)
    # format submission
    submission = ensemble.rank(pct=True, method="first")
    return submission.to_frame("prediction")


#%%


# Quick test
napi.download_dataset(f"{DATA_VERSION}/live.parquet")
live_features = pd.read_parquet(f"{DATA_VERSION}/live.parquet", columns=feature_cols)
predict_ensemble(live_features, benchmark_models)


#%%


# Use the cloudpickle library to serialize your function and its dependencies
import cloudpickle
p = cloudpickle.dumps(predict_ensemble)
with open("target_ensemble.pkl", "wb") as f:
    f.write(p)


#%%


# Download file if running in Google Colab
try:
    from google.colab import files
    files.download('target_ensemble.pkl')
except:
    pass


# That's it! Now head back to [numer.ai](numer.ai) to upload your model!


#%%
# Custom Feature Engineering Functions

def umap_feature_creation(df, features, n_components=50, random_state=42):
    from umap import UMAP
    import pandas as pd
    umap_model = UMAP(n_components=n_components, random_state=random_state)
    umap_feats = umap_model.fit_transform(df[features])
    umap_df = pd.DataFrame(umap_feats, columns=[f'umap_{i}' for i in range(n_components)], index=df.index)
    return pd.concat([df, umap_df], axis=1)


def denoising_autoencoder_features(df, features, encoding_dim=64, noise_factor=0.1, epochs=50, batch_size=256):
    from keras.layers import Input, Dense, GaussianNoise
    from keras.models import Model
    import numpy as np
    import pandas as pd
    
    x = df[features].values
    input_layer = Input(shape=(x.shape[1],))
    noisy = GaussianNoise(noise_factor)(input_layer)
    encoded = Dense(encoding_dim, activation='relu')(noisy)
    decoded = Dense(x.shape[1], activation='linear')(encoded)
    
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    encoder = Model(inputs=input_layer, outputs=encoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(x, x, epochs=epochs, batch_size=batch_size, validation_split=0.1)
    
    encoded_feats = encoder.predict(x)
    enc_df = pd.DataFrame(encoded_feats, columns=[f'ae_{i}' for i in range(encoding_dim)], index=df.index)
    return pd.concat([df, enc_df], axis=1)


def contrastive_feature_creation(df, features, embedding_dim=64, epochs=20, batch_size=256):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import numpy as np
    import pandas as pd

    class TabularDataset(Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            x = self.data[idx]
            view1 = x + np.random.normal(0, 0.01, size=x.shape)
            view2 = x + np.random.normal(0, 0.01, size=x.shape)
            return torch.tensor(view1, dtype=torch.float32), torch.tensor(view2, dtype=torch.float32)

    class Encoder(nn.Module):
        def __init__(self, input_dim, emb_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128), nn.ReLU(),
                nn.Linear(128, emb_dim)
            )
        def forward(self, x):
            return self.net(x)

    class NTXentLoss(nn.Module):
        def __init__(self, temperature=0.5):
            super().__init__()
            self.temperature = temperature
        def forward(self, z1, z2):
            batch_size = z1.shape[0]
            z = torch.cat([z1, z2], dim=0)
            sim = torch.matmul(z, z.T) / self.temperature
            mask = (~torch.eye(2 * batch_size, dtype=bool)).float()
            exp_sim = torch.exp(sim) * mask
            denom = exp_sim.sum(dim=1)
            pos_sim = torch.exp((z1 * z2).sum(dim=1) / self.temperature)
            loss = -torch.log(pos_sim / denom[:batch_size]) - torch.log(pos_sim / denom[batch_size:])
            return loss.mean()

    data = df[features].values
    dataset = TabularDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = Encoder(len(features), embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = NTXentLoss()
    model.train()
    for epoch in range(epochs):
        for v1, v2 in loader:
            optimizer.zero_grad()
            z1 = model(v1)
            z2 = model(v2)
            loss = criterion(z1, z2)
            loss.backward()
            optimizer.step()
    model.eval()
    embeddings = model(torch.tensor(data, dtype=torch.float32)).detach().numpy()
    emb_df = pd.DataFrame(embeddings, columns=[f'ctr_{i}' for i in range(embedding_dim)], index=df.index)
    return pd.concat([df, emb_df], axis=1)


def synthetic_data_ctgan(df, features, epochs=100, batch_size=500):
    from sdv.tabular import CTGAN
    import pandas as pd
    data = df[features]
    ctgan = CTGAN(epochs=epochs, batch_size=batch_size)
    ctgan.fit(data)
    synthetic = ctgan.sample(len(data))
    synthetic.columns = features
    synthetic.columns = [f"{col}_syn" for col in synthetic.columns]
    return pd.concat([df, synthetic], axis=1)

# %%
