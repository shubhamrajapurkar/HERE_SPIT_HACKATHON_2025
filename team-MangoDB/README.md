

# Team MangoDB
Slide pitch deck - [https://www.canva.com/design/DAGdBcQRduY/AM6rFMtkYNS3itm0-p0VGg/edit?utm_content=DAGdBcQRduY&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton]


# Travelo: Smart Multimodal Travel & Emergency Companion

**Team Members:**
Tanvi Patil, Shrihari Mahabal, Deep Patel, Shubham Karampure

---

### **Problem Statement #3: Application Development**

### **Objective**

To build an intelligent mobility assistant that unifies multimodal trip planning with real-time safety responses, leveraging HERE Technologies' APIs and infrastructure.

---

### **Core Features**

#### **1. Unified Multimodal Travel & Ticketing**

* Plan and book trips across buses, metros, trains, cabs, etc.
* One-click ticketing via integrated platform.
* Powered by HERE Public Transit and Routing APIs.

#### **2. AI-Powered Prompt-Based Route Planner**

* Accepts natural language inputs like:
  *“Home via a restaurant stop from college”*
* Uses NLP and Langflow (built on LangChain) to parse travel intents and generate optimized routes.

#### **3. Smart Emergency Response System**

* Crash detection triggers alerts to emergency services.
* Manual incident reports connect users to nearest fire station, hospital, or police station.
* Geolocation and reverse geocoding powered by HERE Location APIs.

---

### **Tech Stack and API Integration**

#### **HERE Technologies APIs Used:**

* **Routing API** – for generating real-time multimodal routes.
* **Public Transit API** – for detailed schedules and transit planning.
* **Geocoding & Search API** – for location lookup and reverse geocoding.
* **Map Tile API** – for rendering interactive maps.
* **HERE MCP Server (Custom Built):**

  * Acts as a middleware between frontend and HERE APIs.
  * Handles API key security, load balancing, and caching.
  * Parses and manages dynamic route configurations and emergency triggers.

---

### **LangGraph NLP Routing Workflow**

1. **Natural Language Input Parsing:**
   Extracts source, destination, and any custom stop requests.

2. **Geocoding Locations:**
   Resolves location names to coordinates using HERE Geocoding API.

3. **Route Mode Determination:**
   Decides between car, public transit, or walking using user preferences and available data.

4. **Multimodal Route Generation:**

   * Public Transit: Routed via OTP API with HERE transit overlays.
   * Car Travel: Uses HERE Routing API with intermediate waypoints.

5. **Final Output:**
   A structured, interactive route plan with real-time guidance.
