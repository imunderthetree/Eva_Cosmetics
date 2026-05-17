# Eva cosmetics Search and Analytics

A team portfolio project that combines a React search interface, a Python search backend, product scraping assets, and reusable analytics scripts for EVA Pharma related datasets.

## Team

- Yusuf Alsaied - 202402431
- Rawan Ibrahim - 202401352

## What This Project Includes

- Product search web app built with React, Vite, TypeScript, and Material UI.
- Python search backend using TF-IDF, query expansion, relevance feedback, and autocomplete suggestions.
- Search index export files for product metadata and TF-IDF vectors.
- Employee email ML analysis for performance scoring and reporting.
- TikTok CSV comparison analysis for engagement summaries and trends.

## Project Flow

1. Product data is collected and stored in `eva_products.csv`.
2. The backend reads the product data, cleans the text, builds searchable documents, and creates a TF-IDF index.
3. User queries are processed with tokenization, query expansion, synonyms, and relevance feedback.
4. The Python API exposes product search, autocomplete suggestions, and a health check.
5. The React frontend sends user searches to the backend and displays ranked product results.
6. Extra analytics scripts can be run separately for employee email analysis and TikTok account comparison.
7. Generated reports are saved under `reports/` and can be regenerated at any time.

## Project Structure

```text
.
|-- analysis/
|   |-- employee_raise_model.py      # Employee email ML pipeline
|   `-- tiktok_comparison.py         # TikTok account comparison
|-- backend/
|   |-- eva_query_processing.py      # Product search and query expansion logic
|   `-- search_api.py                # HTTP API for the React frontend
|-- search_index/                    # Exported product search index assets
|-- selenium/                        # Product scraping source files and scraper
|-- src/                             # React frontend
|-- tiktok scraper/                  # TikTok scraper and CSV exports
|-- eva_group_emails.csv             # Employee email analysis dataset
|-- eva_products.csv                 # Product search dataset
|-- package.json                     # Frontend scripts and dependencies
`-- requirements.txt                 # Python dependencies
```

## Requirements

- Python 3.10+
- Node.js 18+
- npm

Install frontend dependencies:

```bash
npm install
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Run the Web App

Start the Python backend:

```bash
npm run dev:backend
```

In a second terminal, start the frontend:

```bash
npm run dev
```

The frontend runs through Vite and proxies `/api` requests to:

```text
http://127.0.0.1:8000
```

## Product Search Commands

Run a product search from the terminal:

```bash
python backend/eva_query_processing.py search "aloe vera shampoo" --top-k 5
```

Evaluate the search pipeline:

```bash
npm run search:evaluate
```

Export product search assets:

```bash
npm run search:export
```

## Analysis Commands

Run the employee email ML pipeline:

```bash
npm run employee:analyze
```

Run the TikTok comparison:

```bash
npm run tiktok:compare
```

Generated outputs are written to `reports/`. This folder is ignored by Git because reports can be regenerated.

## Backend API

When `npm run dev:backend` is running, the backend exposes:

| Endpoint | Description |
| --- | --- |
| `GET /api/health` | Backend health check and index size |
| `GET /api/search?q=<query>` | Product search with query expansion |
| `GET /api/suggest?q=<prefix>` | Autocomplete suggestions |

## Data and Outputs

Included datasets:

- `eva_products.csv` for product search.
- `eva_group_emails.csv` for the employee email ML demo.
- TikTok CSV exports inside `tiktok scraper/`.

Generated files are intentionally excluded from Git:

- `node_modules/`
- `dist/`
- `reports/`
- Python cache folders
- local database files

## Important Note About the Employee ML Demo

`analysis/employee_raise_model.py` is a portfolio and learning demonstration. It uses engineered signals from sample email data to show a machine learning workflow, including feature engineering, clustering, regression, scoring, and chart generation.

It should not be used as a real HR decision system. Real compensation or promotion decisions require validated criteria, human review, privacy controls, bias testing, and organizational governance.

## Useful Scripts

| Command | Purpose |
| --- | --- |
| `npm run dev` | Start the React frontend |
| `npm run dev:backend` | Start the Python search API |
| `npm run build` | Build the frontend for production |
| `npm run search:evaluate` | Evaluate search quality and latency |
| `npm run search:export` | Export search index assets |
| `npm run employee:analyze` | Run employee email ML analysis |
| `npm run tiktok:compare` | Run TikTok CSV comparison |
