Project -https://github.com/Gideo-git/HERE_SPIT_HACKATHON_2025  
Slide pitch deck -https://www.canva.com/design/DAGjeiv_VQw/hP3sjZIXZiUM6hHyfXAtbw/edit?utm_content=DAGjeiv_VQw&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton
Source code: https://colab.research.google.com/drive/1OGF_bw_DpGL6J661THkvLnEtENleDssK#scrollTo=V9e1ZxNv4P7x



# HERE SPIT Hackathon 2025 — Team CodeBlooded

Welcome to the official repository of **Team CodeBlooded** for the HERE SPIT Hackathon 2025. This project leverages geographic shapefile data to develop innovative solutions for urban navigation and planning.

## 📁 Project Structure

The repository contains the following files and directories:

```
team-CodeBlooded/
│
├── data/
│   └── raw/
│       ├── Streets.cpg
│       ├── Streets.dbf
│       ├── Streets.prj
│       ├── Streets.sbn
│       ├── Streets.sbx
│       ├── Streets.shp
│       └── Streets.shx
│
├── algo.py              # Main algorithm implementation
├── requirements.txt     # Project dependencies
├── .gitignore
└── README.md            # Project documentation
```

The files in `data/raw/` represent the components of a shapefile used for geospatial analysis.

## 🚀 Quick Start

### 🔗 Run in Google Colab (Recommended)

To avoid local setup and dependency issues, we **recommend running this project in Google Colab**:

👉 [Open in Colab](https://colab.research.google.com/drive/1OGF_bw_DpGL6J661THkvLnEtENleDssK#scrollTo=V9e1ZxNv4P7x)

### 🖥️ Run Locally

If you prefer to run the project locally:

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Execute the main script:
   ```bash
   python algo.py
   ```

> **Note:** You will need libraries capable of reading shapefiles (e.g., `geopandas`, `pyshp`) listed in `requirements.txt`.

## 📌 Useful Links

- 🔗 GitHub Repo: [HERE_SPIT_HACKATHON_2025](https://github.com/Gideo-git/HERE_SPIT_HACKATHON_2025)
- 📊 Slide Pitch Deck (Canva): [View Deck](https://www.canva.com/design/DAGjeiv_VQw/hP3sjZIXZiUM6hHyfXAtbw/edit?utm_content=DAGjeiv_VQw&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)
- 📓 Google Colab Notebook: [Open Notebook](https://colab.research.google.com/drive/1OGF_bw_DpGL6J661THkvLnEtENleDssK#scrollTo=V9e1ZxNv4P7x)

## 💡 About

This project integrates HERE Maps data with shapefiles to analyze and enhance urban infrastructure and routing. Developed during the SPIT Hackathon 2025, it showcases the use of spatial data for impactful city planning solutions.

## 👥 Team

Developed by **Team CodeBlooded** for the HERE SPIT Hackathon 2025.
