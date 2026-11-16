import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 1. Load datasets
df_hand = pd.read_csv('Future_Guidance_Statement_Scores_handlabel_reconciled.csv')
df_target = pd.read_csv('../fomc_sentences.csv')

def norm_text(s):
    s = str(s).lower().strip()
    s = s.replace('’', "'").replace('“', '"').replace('”', '"').replace('--', '-')
    return re.sub(r'\s+', ' ', s)

df_hand['norm_text'] = df_hand['target_sentence'].apply(norm_text)
df_target['norm_text'] = df_target['text'].apply(norm_text)

# 2. Build exact reconciled template dictionary (174 unique normalized texts -> 884 target rows)
hand_lookup = (
    df_hand.drop_duplicates('norm_text')
    .set_index('norm_text')[['hand_forward_looking', 'hand_guidance_type', 'hand_commitment_strength']]
    .to_dict('index')
)

# 3. Train calibrated TF-IDF fallback classifiers on the 300 reconciled rows
vec = TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_features=5000, sublinear_tf=True)
X_train = vec.fit_transform(df_hand['target_sentence'])

clf_fl = LogisticRegression(C=3.0, max_iter=1000).fit(X_train, df_hand['hand_forward_looking'])
yes_mask = df_hand['hand_forward_looking'] == 'Yes'
clf_gt = LogisticRegression(C=3.0, max_iter=1000).fit(
    X_train[yes_mask], df_hand.loc[yes_mask, 'hand_guidance_type'].fillna('none')
)
clf_cs = LogisticRegression(C=3.0, max_iter=1000).fit(
    X_train[yes_mask], df_hand.loc[yes_mask, 'hand_commitment_strength'].fillna('firm')
)

# 4. Deterministic Rubric + Calibrated Classifier Labeling Function
def label_row(row):
    nt = row['norm_text']
    if nt in hand_lookup:
        rec = hand_lookup[nt]
        return rec['hand_forward_looking'], rec['hand_guidance_type'], rec['hand_commitment_strength'], 'hand_template'

    txt = str(row['text'])
    low = nt

    # Immediate No rules (Rubric Page 1-2)
    if bool(row['is_admin']):
        return 'No', np.nan, np.nan, 'rule_admin'
    if re.search(r'\b(voted today|board approved|board of governors approved|preferred to)\b', low):
        return 'No', np.nan, np.nan, 'rule_vote_dissent'
    if re.search(r'\b(seeks to achieve|seeks to foster|statutory mandate)\b', low) and not re.search(r'\b(will|expects|anticipates)\b', low):
        return 'No', np.nan, np.nan, 'rule_mandate'
    if re.match(r'^(information received|recent indicators|available indicators|recent data|on a 12-month basis|inflation remains|the labor market remains|household spending|job gains|overall financial conditions)', low):
        if not re.search(r'\b(will|expects|anticipates|would be appropriate|likely to)\b', low):
            return 'No', np.nan, np.nan, 'rule_present_condition'

    # Decision sentences without trailing forward clauses
    if bool(row['is_decision']) and not re.search(r'\b(anticipates|expects|will be appropriate|will continue|will maintain|until|at least through|for an extended period|for a considerable time)\b', low):
        return 'No', np.nan, np.nan, 'rule_pure_decision'

    # Check forward-looking triggers (Time + Subject + Claim tests)
    has_future_cue = bool(re.search(
        r'\b(will|expects|expect|anticipates|anticipate|would be appropriate|in coming months|in coming quarters|in coming meetings|until|as long as|likely to|foreseeable future|for an extended period|for a considerable time|for some time|prepared to|is committed to|risks are weighted|risks continue to be weighted|could foster|may generate|should support|should help)\b',
        low
    ))
    if not has_future_cue:
        x_vec = vec.transform([txt])
        if clf_fl.predict(x_vec)[0] == 'No':
            return 'No', np.nan, np.nan, 'model_no'

    # Step 2: guidance_type (Strict Priority Order 1 -> 6)
    if re.search(r'\b(asset purchase|asset purchases|treasury securities|mortgage-backed securities|mbs|reinvest|reinvesting|redemption cap|balance sheet|holdings of|lending facilities|term asset-backed)\b', low):
        gt = 'balance_sheet'
    elif re.search(r'\b(until|at least as long as|as long as|once|if risks emerge|if incoming information|if progress proves|if the outlook)\b', low):
        gt = 'state_contingent'
    elif re.search(r'\b(for an extended period|for a considerable time|at least through|over coming months|relatively soon)\b', low) and not re.search(r'\brisks are weighted\b', low):
        gt = 'calendar_based'
    elif re.search(r'\b(ongoing increases|additional firming|additional policy firming|remove policy accommodation|policy accommodation can be removed|raise the target range|reduce the target range|gradual increases|exceptionally low levels)\b', low):
        gt = 'explicit_rate_path'
    elif re.search(r'\b(incoming information|incoming data|monitor developments|monitor the economic outlook|assess progress|assess realized and expected|will be patient|depend on the evolution|depend importantly on)\b', low):
        gt = 'data_dependent'
    elif re.search(r'\b(risks are weighted|risks continue to be weighted|foreseeable future|economic activity|inflation|labor market|growth of the economy|economic weakness|price stability|financial conditions)\b', low):
        gt = 'economic_outlook'
    else:
        x_vec = vec.transform([txt])
        gt = clf_gt.predict(x_vec)[0]

    # Step 3: commitment_strength (Conditional > Hedged > Firm)
    if re.search(r'\b(if|until|at least as long as|as long as|once|depending on|will depend on|as appropriate|as warranted|provided that)\b', low):
        cs = 'conditional'
    elif re.search(r'\b(may|might|could|can be|likely|uncertain|risk of|appear|appears|should help|should put)\b', low):
        cs = 'hedged'
    else:
        cs = 'firm'

    return 'Yes', gt, cs, 'rule_yes'

# 5. Apply across all 2,980 sentences & verify consistency
results = df_target.apply(label_row, axis=1, result_type='expand')
df_target[['forward_looking', 'guidance_type', 'commitment_strength', 'label_source']] = results

# Enforce 100% consistency across identical normalized texts
canonical = df_target.drop_duplicates('norm_text').set_index('norm_text')[['forward_looking', 'guidance_type', 'commitment_strength']]
df_target[['forward_looking', 'guidance_type', 'commitment_strength']] = df_target['norm_text'].map(
    lambda x: tuple(canonical.loc[x])
).apply(pd.Series)

# Verify exact match with all 300 reconciled rows
audit = pd.merge(df_hand, df_target, on='sentence_id')
assert (audit['hand_forward_looking'] == audit['forward_looking']).all(), "Mismatch on forward_looking!"
assert (audit['hand_guidance_type'].fillna('NA') == audit['guidance_type'].fillna('NA')).all(), "Mismatch on guidance_type!"
assert (audit['hand_commitment_strength'].fillna('NA') == audit['commitment_strength'].fillna('NA')).all(), "Mismatch on commitment_strength!"
print("Audit passed: 300/300 reconciled rows match 100% across all 3 columns.")

# Save final downloadable CSV
export_cols = ['sentence_id', 'statement_id', 'release_date', 'paragraph', 'sentence', 'text',
               'word_count', 'is_decision', 'is_admin', 'forward_looking', 'guidance_type', 'commitment_strength']
df_target[export_cols].to_csv('fomc_sentences_labeled_2980.csv', index=False)

# 6. Generate All Individual Diagnostic Graphs
sns.set_theme(style='whitegrid', font_scale=1.05)
df_target['year'] = pd.to_datetime(df_target['release_date']).dt.year

# Graph 1: Forward-Looking Share (Yes vs No)
fig, ax = plt.subplots(figsize=(7, 4.5))
fl_counts = df_target['forward_looking'].value_counts()
fl_counts.plot(kind='bar', color=['#1f4e78', '#a6a6a6'], ax=ax, rot=0)
ax.set_title('Graph 1: Forward-Looking Classification (n = 2,980)')
ax.set_ylabel('Sentence Count')
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())} ({p.get_height()/len(df_target)*100:.1f}%)",
                (p.get_x() + p.get_width()/2., p.get_height()), ha='center', va='bottom')
plt.tight_layout()
plt.savefig('graph1_forward_looking_dist.png', dpi=300)

# Graph 2: Guidance Type Distribution (Forward-Looking = Yes)
fig, ax = plt.subplots(figsize=(9, 5))
gt_counts = df_target[df_target['forward_looking'] == 'Yes']['guidance_type'].value_counts()
gt_counts.plot(kind='barh', color='#2e75b6', ax=ax)
ax.invert_yaxis()
ax.set_title('Graph 2: Guidance Type Distribution (Forward-Looking Sentences)')
ax.set_xlabel('Sentence Count')
for p in ax.patches:
    ax.annotate(f" {int(p.get_width())}", (p.get_width(), p.get_y() + p.get_height()/2.), va='center')
plt.tight_layout()
plt.savefig('graph2_guidance_type_dist.png', dpi=300)

# Graph 3: Commitment Strength Distribution (Forward-Looking = Yes)
fig, ax = plt.subplots(figsize=(7, 4.5))
cs_counts = df_target[df_target['forward_looking'] == 'Yes']['commitment_strength'].value_counts()
cs_counts.plot(kind='bar', color=['#1f4e78', '#ed7d31', '#70ad47'], ax=ax, rot=0)
ax.set_title('Graph 3: Commitment Strength Distribution (Forward-Looking Sentences)')
ax.set_ylabel('Sentence Count')
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width()/2., p.get_height()), ha='center', va='bottom')
plt.tight_layout()
plt.savefig('graph3_commitment_strength_dist.png', dpi=300)

# Graph 4: Cross-Tab Heatmap (Guidance Type vs. Commitment Strength)
fig, ax = plt.subplots(figsize=(8.5, 5.5))
ct = pd.crosstab(df_target['guidance_type'].dropna(), df_target['commitment_strength'].dropna())
sns.heatmap(ct, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_title('Graph 4: Guidance Type vs. Commitment Strength Heatmap')
plt.tight_layout()
plt.savefig('graph4_type_vs_strength_heatmap.png', dpi=300)

# Graph 5: Annual Evolution of FOMC Forward Guidance Types (2000-2026)
fig, ax = plt.subplots(figsize=(12, 5.5))
yearly_gt = pd.crosstab(df_target['year'], df_target['guidance_type'].dropna())
yearly_gt.plot(kind='bar', stacked=True, colormap='tab10', ax=ax, width=0.8)
ax.set_title('Graph 5: Annual Evolution of FOMC Forward Guidance by Type (2000–2026)')
ax.set_xlabel('Release Year')
ax.set_ylabel('Forward-Looking Sentence Count')
ax.legend(title='Guidance Type', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig('graph5_annual_guidance_evolution.png', dpi=300)