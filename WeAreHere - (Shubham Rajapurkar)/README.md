# WeAreHere – Roundabout Detection Pipeline 🚦

Welcome to the official repository of *Team WeAreHere* for the *HERE SPIT Hackathon 2025*.  
This project focuses on detecting and validating *roundabouts* using a multi-stage data processing pipeline that combines *graph theory, **geometric validation, and **image processing* techniques.

---

## 🧠 Project Overview

Our goal is to extract accurate roundabout geometries from large road network datasets.  
We begin with a shapefile of 450,000 coordinate points and systematically refine the data through various algorithms to identify approximately *700 valid roundabouts*.

The pipeline includes:
- Road network graph creation
- Cycle detection using NetworkX
- Geometric and spatial filtering
- Image-based circle detection using Hough Transform
- Final output as a .shp file for mapping applications

---

## 📁 Repository Structure

This repository contains three key Jupyter notebooks, each implementing different stages of the pipeline:

### 🔹 Atharva.ipynb
> *Full implementation of Algorithm 2 (Image-Based Detection)*  
This notebook focuses on:
- Rasterizing road geometries to binary images  
- Applying *Canny edge detection* and *Gaussian blur*  
- Detecting circular shapes using *Hough Circle Transform*  
- Mapping detected circles back to geographical coordinates  
- Filtering based on road intersections and geometric constraints

---

### 🔹 gnn.ipynb
> *Graph-based Processing (Algorithm 1)*  
Implements:
- Conversion of the road network into a graph (nodes and edges)
- *Cycle detection* using NetworkX
- *Geometric validation, **compactness, and **aspect ratio* checks
- Road connectivity and *spatial distribution analysis*

---

### 🔹 notebook2990756778.ipynb
> *Preprocessing & Algorithm 1 foundation*  
Handles:
- Reading the input SHP file (450k points)
- Initial data cleaning and transformation  
- Creating the road network graph  
- Preprocessing steps that reduce data volume to 50k points

---

## 📊 Final Output

After running the full pipeline, the final output is a *shapefile (.shp)* containing approximately *700 validated roundabout geometries*, ready for use in:
- Navigation systems  
- Urban planning  
- Smart city mapping applications

---

## 🎥 Presentation

📎 *Project Presentation Link*: (https://www.canva.com/design/DAGoVetRqlA/zCKkmD7Vd4sQwZgn_xIy1g/edit)  
(Google Slides / YouTube / PDF etc.)

---

## 🤝 Team

*Team WeAreHere*  
- Shubham Rajapurkar  
- Atharva [Last Name]  
- [Add other members if applicable]

---

## 🔧 Tech Stack

- Python  
- Jupyter Notebooks  
- NetworkX  
- OpenCV  
- Shapely / Geopandas  
- QGIS (for validation)

---

## 📌 How to Run

1. Clone the repository  
2. Open the notebooks in JupyterLab / Jupyter Notebook  
3. Follow the steps in the order:
    - notebook2990756778.ipynb → gnn.ipynb → Atharva.ipynb  
4. Make sure required packages are installed:
