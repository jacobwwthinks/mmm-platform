# MMM Platform

Multi-client Marketing Mix Modeling for DTC e-commerce brands.

Built for agencies managing brands on Shopify + Meta + Google Ads + TikTok + Pinterest + SMS + Klaviyo.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Windsor.ai API key
export WINDSOR_API_KEY="your-key-here"

# 3. Run the dashboard
streamlit run app.py
```

## How It Works

The platform pulls marketing spend data from Windsor.ai (which aggregates Meta, Google Ads, TikTok, Pinterest, Shopify, Klaviyo, etc.) and fits a Bayesian Marketing Mix Model to decompose revenue into channel contributions.

**What you get:**
- Revenue decomposition (how much each channel contributes)
- ROAS per channel with confidence intervals
- Saturation curves (where are you on the diminishing returns curve?)
- Budget optimization (where should you shift spend?)
- Support for new vs. returning customer revenue split

## Adding a New Client

1. Connect their ad platforms and Shopify in Windsor.ai
2. Add their account IDs to `config.yaml`:

```yaml
new_client:
  display_name: "Brand Name"
  currency: "SEK"
  channels:
    meta:
      windsor_connector: "facebook"
      windsor_account: "YOUR_ACCOUNT_ID"
      spend_field: "spend"
      impressions_field: "impressions"
    google_ads:
      windsor_connector: "google_ads"
      windsor_account: "YOUR_ACCOUNT_ID"
      spend_field: "spend"
      impressions_field: "impressions"
  revenue_source:
    windsor_connector: "shopify"
    windsor_account: "store.myshopify.com"
    revenue_field: "order_total_price"
    orders_field: "order_count"
  events_csv: "events/new_client_events.csv"
```

3. Create an event calendar CSV (or generate a template from the Event Calendar page)
4. Run the model from the dashboard

## Event Calendar

Each client needs an event calendar CSV tracking promotions and product drops. Columns:

| Column | Type | Description |
|--------|------|-------------|
| week_start | date | Monday of the week (YYYY-MM-DD) |
| discount_campaign | 0/1 | Sale or discount active this week |
| product_drop | 0/1 | New product launched this week |
| holiday | 0/1 | Major shopping holiday (BF, Xmas, etc.) |
| notes | text | Description (optional) |

The app can auto-generate a template with holidays pre-filled.

## SMS & Email Data

- **Klaviyo (email):** Connect in Windsor.ai, add account ID to config
- **SMS:** Upload a CSV with `week_start` and `spend` columns via the Event Calendar page

## New vs. Returning Customer Revenue

The model can target either total revenue, new customer revenue, or returning customer revenue. For evaluating paid media effectiveness, **new customer revenue** is recommended since paid ads primarily drive new customer acquisition. Returning customer revenue is better explained by email, SMS, and promotional events.

## Deployment

**Option 1: Streamlit Community Cloud (free, easiest)**
1. Push this project to a GitHub repo
2. Go to share.streamlit.io
3. Connect your repo and set `WINDSOR_API_KEY` as a secret

**Option 2: Any server**
```bash
pip install -r requirements.txt
export WINDSOR_API_KEY="your-key"
streamlit run app.py --server.port 8501
```

**Password protection:** Add to `.streamlit/secrets.toml`:
```toml
[passwords]
team_password = "your-password"
```

## Model Details

The MMM uses Bayesian estimation with:
- **Geometric adstock** for carryover effects (ads seen this week still affect next week)
- **Hill saturation** for diminishing returns (more spend = less incremental effect)
- **Fourier seasonality** for weekly/monthly patterns
- **Bootstrap uncertainty** for confidence intervals on all estimates

Channel priors are calibrated for DTC e-commerce based on industry benchmarks. The model runs on CPU in 1-5 minutes per client.

## Project Structure

```
mmm-platform/
├── app.py                  # Streamlit entry point
├── config.yaml             # Client configuration
├── requirements.txt
├── data/
│   ├── ingest.py           # Windsor.ai data pulling
│   ├── process.py          # Data cleaning & aggregation
│   └── events.py           # Event calendar management
├── model/
│   ├── mmm.py              # Core MMM model
│   ├── priors.py           # DTC channel priors
│   └── diagnostics.py      # Model quality checks
├── optimize/
│   └── budget.py           # Budget allocation optimizer
├── pages/
│   ├── 1_Client_Overview.py
│   ├── 2_Channel_Analysis.py
│   ├── 3_Budget_Optimizer.py
│   └── 4_Event_Calendar.py
└── events/                 # Client event calendar CSVs
```
