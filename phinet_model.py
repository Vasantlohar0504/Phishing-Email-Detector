import pandas as pd
import numpy as np
import re
import tldextract
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_X_y, check_array

# ----- Feature Engineering -----
class PHINetFeatureEngine:
    keyword_list = ['login', 'verify', 'bank', 'account', 'urgent', 'click', 'password', 'update']
    trusted_tlds = ['com', 'org', 'net']
    trusted_brands = ['paypal', 'bankofamerica', 'chase', 'apple', 'amazon']

    def __init__(self):
        self.domain_encoder = LabelEncoder()
        self.fitted_domains = []

    def has_deceptive_domain(self, url):
        url = str(url) if pd.notnull(url) else ""
        ext = tldextract.extract(url)
        for brand in self.trusted_brands:
            if brand in url.lower():
                if ext.domain != brand or ext.suffix not in self.trusted_tlds:
                    return 1
        return 0

    def suspicious_subdomain(self, url):
        url = str(url) if pd.notnull(url) else ""
        ext = tldextract.extract(url)
        sub_parts = ext.subdomain.split('.') if ext.subdomain else []
        return int(len(sub_parts) > 2 or any(len(p) > 15 for p in sub_parts))

    def sender_mismatch(self, domain):
        domain = str(domain).lower()
        for brand in self.trusted_brands:
            if brand in domain and not domain.endswith(f"{brand}.com"):
                return 1
        return 0

    def transform(self, df):
        df = df.copy()
        df['body_length'] = df['email_body'].apply(lambda x: len(str(x)))
        df['suspicious_keywords'] = df['email_body'].apply(
            lambda body: sum(kw in str(body).lower() for kw in self.keyword_list)
        )
        df['url_count'] = df['urls'].apply(lambda u: len(re.findall(r'https?://', str(u))))
        df['https_count'] = df['urls'].apply(lambda u: str(u).count("https://"))
        df['attachment_risk'] = df['attachments'].apply(
            lambda att: int(any(ext in str(att).lower() for ext in ['.exe', '.zip', '.bat', '.scr', '.rar']))
        )
        df['sender_domain'] = df['email_id'].apply(lambda e: e.split('@')[-1] if '@' in str(e) else 'unknown')

        if not self.fitted_domains:
            self.fitted_domains = df['sender_domain'].unique().tolist()
            if 'unknown' not in self.fitted_domains:
                self.fitted_domains.append('unknown')
            self.domain_encoder.fit(self.fitted_domains)

        df['sender_domain_encoded'] = df['sender_domain'].apply(
            lambda x: x if x in self.fitted_domains else 'unknown')

        df['sender_domain_encoded'] = self.domain_encoder.transform(df['sender_domain_encoded'])

        df['deceptive_url_flag'] = df['urls'].apply(self.has_deceptive_domain)
        df['subdomain_depth_flag'] = df['urls'].apply(self.suspicious_subdomain)
        df['sender_mismatch_flag'] = df['sender_domain'].apply(self.sender_mismatch)

        return df[[
            'body_length',
            'suspicious_keywords',
            'url_count',
            'https_count',
            'attachment_risk',
            'sender_domain_encoded',
            'deceptive_url_flag',
            'subdomain_depth_flag',
            'sender_mismatch_flag'
        ]]

# ----- Boosting Model -----
class PHINetBoost(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=7, max_depth=3):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.models = []

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        residual = y.copy()
        self.models = []
        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(X, residual)
            pred = tree.predict(X)
            residual = y - pred
            self.models.append(tree)
        return self

    def predict(self, X):
        X = check_array(X)
        total_preds = np.sum([model.predict(X) for model in self.models], axis=0)
        return (total_preds >= (self.n_estimators / 2)).astype(int)

# ----- Training Wrapper -----
def train_phinet_model(df):
    feature_engine = PHINetFeatureEngine()
    X = feature_engine.transform(df)
    y = df['label'].astype(int)
    model = PHINetBoost(n_estimators=7)
    model.fit(X, y)
    return model, feature_engine