import React, { useState, useCallback } from 'react';
import { MapPin, Navigation, TrendingUp, AlertTriangle, Home, Upload, BarChart3, Clock, Users, Building, Shield, Car } from 'lucide-react';

const GeoIntelligenceDashboard = () => {
  const [geoJsonData, setGeoJsonData] = useState(null);
  const [analysisResults, setAnalysisResults] = useState({
    emergencyResponse: {
      hospitals: [
        {
          name: "Kokilaben Dhirubhai Ambani Hospital",
          address: "Rao Saheb Achutrao Patwardhan Marg, Four Bungalows, Andheri West, Mumbai, Maharashtra 400053",
          coordinates: [19.1364, 72.8296],
          responseTime: "8 minutes",
          capacity: "750 beds",
          equipment: ["MRI", "CT Scan", "ICU", "Emergency"]
        },
        {
          name: "Lilavati Hospital",
          address: "A-791, Bandra Reclamation, Bandra West, Mumbai, Maharashtra 400050",
          coordinates: [19.0665, 72.8314],
          responseTime: "10 minutes",
          capacity: "500 beds",
          equipment: ["Cardiology", "Neurology", "Oncology"]
        },
        {
          name: "Bombay Hospital",
          address: "12, New Marine Lines, Mumbai, Maharashtra 400020",
          coordinates: [18.9439, 72.8259],
          responseTime: "12 minutes",
          capacity: "800 beds",
          equipment: ["Trauma Center", "Burn Unit", "Pediatrics"]
        },
        {
          name: "Sion Hospital",
          address: "Dr. Babasaheb Ambedkar Rd, Sion West, Sion, Mumbai, Maharashtra 400022",
          coordinates: [19.0390, 72.8600],
          responseTime: "15 minutes",
          capacity: "1200 beds",
          equipment: ["General Medicine", "Surgery", "Orthopedics"]
        }
      ],
      routeOptimization: "Optimal routes calculated based on hospital locations and current traffic conditions",
      recommendations: [
        "Establish additional trauma centers in eastern suburbs",
        "Improve ambulance response times during peak hours",
        "Coordinate with traffic police for emergency corridors"
      ]
    },
    trafficPrediction: {
      hotspots: [
        {
          location: "Andheri Flyover",
          address: "Western Express Highway, Andheri East, Mumbai",
          coordinates: [19.1197, 72.8464],
          peakTime: "08:00-11:00, 18:00-21:00",
          severity: "High",
          impact: "Delays of 45-60 minutes expected"
        },
        {
          location: "Sion Circle",
          address: "Sion Circle, Sion, Mumbai",
          coordinates: [19.0390, 72.8600],
          peakTime: "07:30-10:30, 17:30-20:30",
          severity: "Medium",
          impact: "Delays of 30-45 minutes expected"
        },
        {
          location: "Dadar TT Circle",
          address: "Dadar West, Mumbai",
          coordinates: [19.0199, 72.8436],
          peakTime: "All day",
          severity: "High",
          impact: "Constant congestion with 30+ minute delays"
        },
        {
          location: "Worli Sea Face",
          address: "Worli Sea Face, Worli, Mumbai",
          coordinates: [19.0160, 72.8215],
          peakTime: "17:00-20:00",
          severity: "Medium",
          impact: "Weekend traffic with 20-30 minute delays"
        }
      ],
      insights: "Major congestion points identified at key intersections and flyovers. Monsoon season exacerbates traffic due to waterlogging.",
      solutions: [
        "Implement smart traffic signals with AI coordination",
        "Expand public transport options during peak hours",
        "Create alternative routes for emergency vehicles"
      ]
    },
    businessOpportunities: {
      underservedAreas: [
        {
          area: "Govandi",
          address: "Govandi East, Mumbai",
          coordinates: [19.0569, 72.9086],
          missingServices: ["Quality healthcare", "Shopping malls", "Premium education"],
          opportunity: "High",
          population: "500,000+"
        },
        {
          area: "Mankhurd",
          address: "Mankhurd, Mumbai",
          coordinates: [19.0500, 72.9300],
          missingServices: ["Supermarkets", "Entertainment", "Banking"],
          opportunity: "High",
          population: "300,000+"
        },
        {
          area: "Bhandup West",
          address: "Bhandup West, Mumbai",
          coordinates: [19.1400, 72.9300],
          missingServices: ["Specialty hospitals", "Fine dining", "Tech hubs"],
          opportunity: "Medium",
          population: "200,000+"
        },
        {
          area: "Kanjurmarg",
          address: "Kanjurmarg East, Mumbai",
          coordinates: [19.1300, 72.9400],
          missingServices: ["Premium housing", "IT parks", "International schools"],
          opportunity: "Medium",
          population: "150,000+"
        }
      ],
      marketValue: "Eastern suburbs show high potential for development with growing population",
      recommendations: [
        "Invest in mixed-use developments in Govandi-Mankhurd belt",
        "Establish healthcare facilities in underserved areas",
        "Develop IT parks along metro corridors"
      ]
    },
    disasterManagement: {
      floodZones: [
        {
          area: "Chembur Low-lying Areas",
          address: "Chembur, Mumbai",
          coordinates: [19.0600, 72.9000],
          riskLevel: "High",
          elevation: "2 meters",
          shelters: [
            {
              name: "Chembur Welfare Association Hall",
              address: "Near Chembur Station, Mumbai",
              capacity: "200 people"
            }
          ]
        },
        {
          area: "Kurla West",
          address: "Kurla West, Mumbai",
          coordinates: [19.0750, 72.8800],
          riskLevel: "High",
          elevation: "3 meters",
          shelters: [
            {
              name: "Kurla Community Center",
              address: "LBS Marg, Kurla West",
              capacity: "300 people"
            }
          ]
        },
        {
          area: "Dharavi",
          address: "Dharavi, Mumbai",
          coordinates: [19.0400, 72.8500],
          riskLevel: "Very High",
          elevation: "1 meter",
          shelters: [
            {
              name: "Dharavi Municipal School",
              address: "90 Feet Road, Dharavi",
              capacity: "500 people"
            }
          ]
        },
        {
          area: "Mahim Creek Area",
          address: "Mahim West, Mumbai",
          coordinates: [19.0350, 72.8400],
          riskLevel: "Medium",
          elevation: "4 meters",
          shelters: [
            {
              name: "Mahim Church Hall",
              address: "St. Michael's Church, Mahim",
              capacity: "150 people"
            }
          ]
        }
      ],
      evacuationTime: "30-60 minutes depending on water levels and population density",
      preparedness: [
        "Install early warning systems in flood-prone areas",
        "Conduct regular evacuation drills before monsoon",
        "Stock emergency supplies at all shelter locations"
      ]
    },
    neighborhoodHealth: {
      rankings: [
        {
          area: "Bandra West",
          address: "Bandra West, Mumbai",
          coordinates: [19.0550, 72.8300],
          score: "9/10",
          strengths: ["Excellent healthcare", "Good schools", "Recreational spaces"],
          weaknesses: ["High cost of living", "Traffic congestion"]
        },
        {
          area: "Juhu",
          address: "Juhu, Mumbai",
          coordinates: [19.1075, 72.8263],
          score: "8/10",
          strengths: ["Beach access", "Entertainment options", "Safety"],
          weaknesses: ["Flood risk", "Parking issues"]
        },
        {
          area: "Powai",
          address: "Powai, Mumbai",
          coordinates: [19.1200, 72.9100],
          score: "7/10",
          strengths: ["IT hub", "Good infrastructure", "Educational institutes"],
          weaknesses: ["Distance from city center", "Limited public transport"]
        },
        {
          area: "Ghatkopar",
          address: "Ghatkopar East, Mumbai",
          coordinates: [19.0800, 72.9100],
          score: "6/10",
          strengths: ["Metro connectivity", "Affordable housing"],
          weaknesses: ["Overcrowding", "Pollution"]
        }
      ],
      demographics: "Western suburbs score higher due to better infrastructure and amenities",
      insights: [
        "Coastal areas have better amenities but higher flood risk",
        "Eastern suburbs need infrastructure improvements",
        "Metro connectivity is improving neighborhood scores"
      ]
    }
  });
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);

  const processGeoJsonData = useCallback(async (data) => {
    setIsProcessing(true);
    setError(null);
    
    try {
      const geoJsonString = JSON.stringify(data, null, 2);
      
      const prompt = `Analyze this GeoJSON data from Mumbai/Navi Mumbai/Thane region and provide detailed insights including exact locations and addresses for each point. Focus on these aspects:

1. Emergency Response: Identify hospitals with names, exact addresses, coordinates, capacity
2. Traffic Prediction: Highlight congestion points with street names, landmarks, peak times
3. Business Opportunities: Show underserved areas with precise locations and missing services
4. Disaster Management: Pinpoint flood zones with neighborhood names and risk levels
5. Neighborhood Health: Rank areas with addresses and service proximity details

Return structured JSON with all location details included in each section. Here's the GeoJSON data:

${geoJsonString}

Format your response as valid JSON with these sections:
{
  "emergencyResponse": {
    "hospitals": [{
      "name": "Hospital Name",
      "address": "Full address with street",
      "coordinates": [lat, lng],

    }],
    "routeOptimization": "Text description",
    "recommendations": ["List"]
  },
  "trafficPrediction": {
    "hotspots": [{
      "location": "Intersection/Street name",
      "address": "Full address",
      "coordinates": [lat, lng],
      "peakTime": "HH:MM-HH:MM",
      "severity": "High/Medium/Low",
      "impact": "Text"
    }],
    "insights": "Text description",
    "solutions": ["List"]
  },
  "businessOpportunities": {
    "underservedAreas": [{
      "area": "Neighborhood name",
      "address": "Full address",
      "coordinates": [lat, lng],

    }],
    "marketValue": "Text description",
    "recommendations": ["List"]
  },
  "disasterManagement": {
    "floodZones": [{
      "area": "Zone name",
      "address": "Full address",
      "coordinates": [lat, lng],
      "riskLevel": "High/Medium/Low",
      "elevation": "X meters",
      "shelters": [{
        "name": "Shelter name",
        "address": "Full address",
        "capacity": "X people"
      }]
    }],
    "evacuationTime": "Text description",
    "preparedness": ["List"]
  },
  "neighborhoodHealth": {
    "rankings": [{
      "area": "Neighborhood name",
      "address": "Full address",
      "coordinates": [lat, lng],
      "score": "X/10",
      "strengths": ["List"],
      "weaknesses": ["List"]
    }],
    "demographics": "Text description",
    "insights": ["List"]
  }
}`;

      const response = await fetch(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=AIzaSyBXvyQXa7LjTNqqDkm3uvubhhkQ1A5dWZs",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            contents: [{ parts: [{ text: prompt }] }],
          }),
        }
      );

      if (!response.ok) {
        throw new Error(`API request failed: ${response.status} ${response.statusText}`);
      }

      const result = await response.json();
      const geminiResponse = result.candidates[0].content.parts[0].text;
      
      // Try to parse the JSON response
      let parsedResults;
      try {
        const cleanedResponse = geminiResponse.replace(/```json\n?|\n?```/g, '').trim();
        parsedResults = JSON.parse(cleanedResponse);
        
        // Validate the structure has at least some data
        if (!parsedResults.emergencyResponse && 
            !parsedResults.trafficPrediction && 
            !parsedResults.businessOpportunities && 
            !parsedResults.disasterManagement && 
            !parsedResults.neighborhoodHealth) {
          throw new Error("Response doesn't contain expected analysis sections");
        }
      } catch (parseError) {
        console.error('Failed to parse JSON response:', parseError);
        // Create a more detailed fallback response
        parsedResults = {
          emergencyResponse: {
            hospitals: data.features
              .filter(f => f.properties?.name?.toLowerCase().includes('hospital'))
              .map(h => ({
                name: h.properties.name || 'Unknown Hospital',
                address: h.properties.address || 'Address not available',
                coordinates: h.geometry.coordinates,
                responseTime: "5-15 minutes",
                capacity: "50-200 beds",
                equipment: ["Emergency", "ICU", "Surgery"]
              })),
            routeOptimization: "Routes calculated based on hospital locations in the GeoJSON data",
            recommendations: [
              "Prioritize hospitals with emergency facilities",
              "Consider traffic patterns when routing ambulances",
              "Update hospital capacity data regularly"
            ]
          },
          trafficPrediction: {
            hotspots: data.features
              .filter(f => f.properties?.name?.toLowerCase().includes('road') || 
                          f.properties?.name?.toLowerCase().includes('intersection'))
              .map(h => ({
                location: h.properties.name || 'Major Intersection',
                address: h.properties.address || 'Address not available',
                coordinates: h.geometry.coordinates,
                peakTime: "08:00-10:00, 17:00-19:00",
                severity: "High",
                impact: "30-45 minute delays expected"
              })),
            insights: "Traffic patterns identified from road network in the GeoJSON data",
            solutions: [
              "Implement alternate routes during peak hours",
              "Add traffic signals at busy intersections",
              "Improve public transport options"
            ]
          },
          businessOpportunities: {
            underservedAreas: data.features
              .filter(f => f.properties?.weight < 10) // Example filter
              .map(a => ({
                area: a.properties.name || 'Commercial Area',
                address: a.properties.address || 'Address not available',
                coordinates: a.geometry.coordinates,
                missingServices: ["Pharmacies", "Supermarkets", "Clinics"],
                opportunity: "High",
                population: "10,000-50,000"
              })),
            marketValue: "High potential areas identified from commercial locations in GeoJSON",
            recommendations: [
              "Open essential services in underserved areas",
              "Focus on high-density residential zones",
              "Consider local demographics when planning"
            ]
          },
          disasterManagement: {
            floodZones: data.features
              .filter(f => f.geometry.type === 'Polygon') // Example filter
              .map(z => ({
                area: z.properties.name || 'Low-lying Area',
                address: z.properties.address || 'Address not available',
                coordinates: z.geometry.coordinates[0][0], // First point of polygon
                riskLevel: "High",
                elevation: "2-5 meters",
                shelters: [{
                  name: "Nearest Emergency Shelter",
                  address: "Address not specified",
                  capacity: "100-500 people"
                }]
              })),
            evacuationTime: "15-30 minutes depending on location",
            preparedness: [
              "Identify multiple evacuation routes",
              "Mark safe zones clearly",
              "Conduct regular drills"
            ]
          },
          neighborhoodHealth: {
            rankings: data.features
              .filter(f => f.properties?.confidence_percent > 90) // Example filter
              .map(n => ({
                area: n.properties.name || 'Residential Area',
                address: n.properties.address || 'Address not available',
                coordinates: n.geometry.coordinates,
                score: "7-9/10",
                strengths: ["Good transport", "Nearby markets", "Schools"],
                weaknesses: ["Lacking hospitals", "Few parks", "Poor drainage"]
              })),
            demographics: "Mixed residential and commercial areas in the GeoJSON data",
            insights: [
              "Higher scores near commercial centers",
              "Lower scores in peripheral areas",
              "Good overall service distribution"
            ]
          },
          rawGeminiResponse: geminiResponse
        };
      }
      
      setAnalysisResults(parsedResults);
    } catch (error) {
      console.error('Error processing GeoJSON data:', error);
      setError(`Analysis failed: ${error.message}. Try uploading a different GeoJSON file.`);
    } finally {
      setIsProcessing(false);
    }
  }, []);

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const jsonData = JSON.parse(e.target.result);
          setGeoJsonData(jsonData);
          processGeoJsonData(jsonData);
        } catch (error) {
          setError('Invalid GeoJSON file. Please upload a valid JSON file.');
        }
      };
      reader.readAsText(file);
    }
  };

  const InsightCard = ({ title, icon: Icon, children, color = "blue" }) => {
    const colorClasses = {
      blue: "border-blue-200 bg-blue-50",
      red: "border-red-200 bg-red-50",
      green: "border-green-200 bg-green-50",
      yellow: "border-yellow-200 bg-yellow-50",
      purple: "border-purple-200 bg-purple-50"
    };

    const iconColors = {
      blue: "text-blue-600",
      red: "text-red-600",
      green: "text-green-600",
      yellow: "text-yellow-600",
      purple: "text-purple-600"
    };

    return (
      <div className={`border-2 rounded-lg p-6 ${colorClasses[color]} shadow-sm`}>
        <div className="flex items-center mb-4">
          <Icon className={`w-6 h-6 mr-3 ${iconColors[color]}`} />
          <h2 className="text-xl font-bold text-gray-800">{title}</h2>
        </div>
        {children}
      </div>
    );
  };

  const MetricCard = ({ label, value, sublabel, color = "gray" }) => {
    const colorClasses = {
      gray: "text-gray-600",
      red: "text-red-600",
      yellow: "text-yellow-600",
      green: "text-green-600",
      blue: "text-blue-600",
      purple: "text-purple-600"
    };

    return (
      <div className="bg-white border rounded-lg p-4 shadow-sm">
        <div className="text-sm text-gray-600 mb-1">{label}</div>
        <div className={`text-2xl font-bold ${colorClasses[color]} mb-1`}>{value}</div>
        {sublabel && <div className="text-xs text-gray-500">{sublabel}</div>}
      </div>
    );
  };

  const DataTable = ({ data, columns }) => {
    if (!data || !Array.isArray(data) || data.length === 0) {
      return (
        <div className="bg-gray-50 p-4 rounded-lg text-center text-gray-500">
          No location data available for this analysis.
        </div>
      );
    }

    // Get all columns from the first data item if specific columns aren't provided
    const tableColumns = columns || Object.keys(data[0] || {});

    // Function to format coordinates for display
    const formatValue = (value) => {
      if (Array.isArray(value) && value.length === 2 && typeof value[0] === 'number') {
        return `[${value[0].toFixed(6)}, ${value[1].toFixed(6)}]`;
      }
      if (typeof value === 'object' && value !== null) {
        return JSON.stringify(value);
      }
      return String(value);
    };

    return (
      <div className="overflow-x-auto">
        <table className="w-full bg-white border border-gray-200 rounded-lg">
          <thead className="bg-gray-50">
            <tr>
              {tableColumns.map((col, idx) => (
                <th key={idx} className="px-4 py-3 text-left text-sm font-semibold text-gray-700 border-b">
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.map((row, idx) => (
              <tr key={idx} className="border-b hover:bg-gray-50">
                {tableColumns.map((col, cellIdx) => (
                  <td key={cellIdx} className="px-4 py-3 text-sm text-gray-700">
                    {row[col] !== undefined ? formatValue(row[col]) : '-'}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-4">
            Real-Time Geo-Intelligence Dashboard
          </h1>
          <p className="text-lg text-gray-600 mb-6">
            AI-Powered Urban Analytics for Mumbai Metropolitan Region using Gemini AI
          </p>
          
          {/* {!geoJsonData && (
            <div className="bg-white border-2 border-dashed border-gray-300 rounded-lg p-8">
              <Upload className="w-12 h-12 mx-auto mb-4 text-gray-400" />
              <label className="cursor-pointer">
                <span className="text-lg font-medium text-blue-600 hover:text-blue-500">
                  Upload GeoJSON File
                </span>
                <input
                  type="file"
                  accept=".json,.geojson"
                  onChange={handleFileUpload}
                  className="hidden"
                />
              </label>
              <p className="text-sm text-gray-500 mt-2">
                Upload Mumbai, Navi Mumbai, or Thane GeoJSON data to begin real-time AI analysis
              </p>
            </div>
          )} */}

          {error && (
            <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
              <strong>Error:</strong> {error}
            </div>
          )}
        </div>

        {isProcessing && (
          <div className="text-center py-12">
            <BarChart3 className="w-16 h-16 mx-auto mb-4 text-blue-600 animate-spin" />
            <h2 className="text-xl font-semibold text-gray-700 mb-2">Processing GeoJSON Data with Gemini AI</h2>
            <p className="text-gray-600">Analyzing spatial patterns and generating real-time insights...</p>
          </div>
        )}

        {analysisResults && (
          <div className="space-y-8">
            {/* Show raw Gemini response if JSON parsing failed */}
            {analysisResults.rawGeminiResponse && (
              <div className="bg-white border rounded-lg p-6 shadow-sm">
                <h2 className="text-xl font-bold text-gray-800 mb-4">Gemini AI Analysis Results</h2>
                <div className="bg-gray-50 p-4 rounded-lg">
                  <pre className="text-sm text-gray-700 whitespace-pre-wrap">{analysisResults.rawGeminiResponse}</pre>
                </div>
              </div>
            )}

            {/* Emergency Response System */}
            <InsightCard title="Smart Emergency Response System" icon={Shield} color="red">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <MetricCard label="Analysis Status" value="Active" sublabel="Real-time processing" color="red" />
                <MetricCard label="Data Source" value="GeoJSON" sublabel="Spatial coordinates" color="red" />
                <MetricCard label="AI Engine" value="Gemini" sublabel="Google AI analysis" color="red" />
              </div>
              
              <h3 className="text-lg font-semibold mb-3">Emergency Response Analysis</h3>
              <DataTable 
                data={analysisResults.emergencyResponse.hospitals}
                columns={["name", "address", "responseTime", "capacity"]}
              />
              
              <div className="mt-4">
                <h4 className="font-semibold mb-2">AI-Generated Recommendations</h4>
                <ul className="space-y-1">
                  {analysisResults.emergencyResponse.recommendations.map((rec, idx) => (
                    <li key={idx} className="text-sm text-gray-700 flex items-start">
                      <span className="text-red-500 mr-2">•</span>
                      {rec}
                    </li>
                  ))}
                </ul>
              </div>
            </InsightCard>

            {/* Traffic Prediction */}
            <InsightCard title="Traffic Chaos Predictor" icon={Car} color="yellow">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <MetricCard label="Hotspots Detected" value="4" sublabel="Major locations" color="yellow" />
                <MetricCard label="Prediction Engine" value="Gemini AI" sublabel="Machine learning" color="yellow" />
                <MetricCard label="Peak Hours" value="8-11AM, 6-9PM" sublabel="High congestion" color="yellow" />
              </div>
              
              <h3 className="text-lg font-semibold mb-3">Traffic Hotspot Analysis</h3>
              <DataTable 
                data={analysisResults.trafficPrediction.hotspots}
                columns={["location", "address", "peakTime", "severity"]}
              />
              
              <div className="mt-4 p-4 bg-yellow-100 rounded-lg">
                <h4 className="font-semibold mb-2">Gemini AI Traffic Insights</h4>
                <p className="text-sm text-gray-700">{analysisResults.trafficPrediction.insights}</p>
              </div>
            </InsightCard>

            {/* Business Opportunities */}
            <InsightCard title="Business Goldmine Detector" icon={TrendingUp} color="green">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <MetricCard label="Underserved Areas" value="4" sublabel="High potential" color="green" />
                <MetricCard label="Opportunity Level" value="High" sublabel="Eastern suburbs" color="green" />
                <MetricCard label="Population" value="1M+" sublabel="Target market" color="green" />
              </div>
              
              <h3 className="text-lg font-semibold mb-3">Market Opportunity Analysis</h3>
              <DataTable 
                data={analysisResults.businessOpportunities.underservedAreas}
                columns={["area", "address", "missingServices", "opportunity"]}
              />
              
              <div className="mt-4">
                <h4 className="font-semibold mb-2">AI Investment Insights</h4>
                <ul className="space-y-1">
                  {analysisResults.businessOpportunities.recommendations.map((rec, idx) => (
                    <li key={idx} className="text-sm text-gray-700 flex items-start">
                      <span className="text-green-500 mr-2">•</span>
                      {rec}
                    </li>
                  ))}
                </ul>
              </div>
            </InsightCard>

            {/* Disaster Management */}
            <InsightCard title="Flood & Disaster Escape Planner" icon={AlertTriangle} color="blue">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <MetricCard label="High Risk Zones" value="3" sublabel="Flood-prone" color="blue" />
                <MetricCard label="Shelters" value="4" sublabel="Emergency locations" color="blue" />
                <MetricCard label="Evacuation Time" value="30-60 mins" sublabel="Estimated" color="blue" />
              </div>
              
              <h3 className="text-lg font-semibold mb-3">Disaster Risk Analysis</h3>
              <DataTable 
                data={analysisResults.disasterManagement.floodZones}
                columns={["area", "address", "riskLevel", "elevation"]}
              />
              
              <div className="mt-4 p-4 bg-blue-100 rounded-lg">
                <h4 className="font-semibold mb-2">AI Disaster Preparedness</h4>
                <ul className="space-y-1">
                  {analysisResults.disasterManagement.preparedness.map((item, idx) => (
                    <li key={idx} className="text-sm text-gray-700 flex items-start">
                      <span className="text-blue-500 mr-2">•</span>
                      {item}
                    </li>
                  ))}
                </ul>
              </div>
            </InsightCard>

            {/* Neighborhood Health */}
            <InsightCard title="Neighborhood Health Scanner" icon={Home} color="purple">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <MetricCard label="Areas Analyzed" value="4" sublabel="Key neighborhoods" color="purple" />
                <MetricCard label="Top Score" value="9/10" sublabel="Bandra West" color="purple" />
                <MetricCard label="Lowest Score" value="6/10" sublabel="Ghatkopar" color="purple" />
              </div>
              
              <h3 className="text-lg font-semibold mb-3">Neighborhood Health Rankings</h3>
              <DataTable 
                data={analysisResults.neighborhoodHealth.rankings}
                columns={["area", "address", "score", "strengths"]}
              />
              
              <div className="mt-4">
                <h4 className="font-semibold mb-2">Gemini AI Urban Insights</h4>
                <ul className="space-y-1">
                  {analysisResults.neighborhoodHealth.insights.map((insight, idx) => (
                    <li key={idx} className="text-sm text-gray-700 flex items-start">
                      <span className="text-purple-500 mr-2">•</span>
                      {insight}
                    </li>
                  ))}
                </ul>
              </div>
            </InsightCard>

            {/* AI Analysis Summary */}
            <div className="bg-white border rounded-lg p-6 shadow-sm">
              <div className="flex items-center mb-4">
                <BarChart3 className="w-6 h-6 mr-3 text-gray-600" />
                <h2 className="text-xl font-bold text-gray-800">Mumbai Geo-Intelligence Summary</h2>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <MetricCard label="Hospitals Mapped" value="4" sublabel="Emergency response" />
                <MetricCard label="Traffic Hotspots" value="4" sublabel="Congestion points" />
                <MetricCard label="Business Areas" value="4" sublabel="High potential" />
                <MetricCard label="Flood Zones" value="4" sublabel="Risk areas" />
              </div>
              <div className="mt-4 p-4 bg-gray-50 rounded-lg">
                <p className="text-sm text-gray-700">
                  This geo-intelligence dashboard displays hardcoded Mumbai data for demonstration purposes. 
                  The data includes key locations across emergency services, traffic patterns, business opportunities, 
                  disaster zones, and neighborhood health metrics. In a production environment, this would be connected 
                  to real-time GeoJSON data feeds and processed through Gemini AI for dynamic analysis.
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default GeoIntelligenceDashboard;