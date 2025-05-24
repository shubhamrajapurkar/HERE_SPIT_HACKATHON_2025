import { useState, useEffect, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Textarea } from '@/components/ui/textarea';
import { ArrowLeft, Upload, MapPin, BarChart3, Eye, Download, Settings } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

// Leaflet imports (we'll add this as a dependency)
declare global {
  interface Window {
    L: any;
  }
}

interface CSVMapPlotterProps {
  onBack: () => void;
}

const CSVMapPlotter = ({ onBack }: CSVMapPlotterProps) => {
  const [csvData, setCsvData] = useState(``);
  
  const [totalPoints, setTotalPoints] = useState(0);
  const [plottedPoints, setPlottedPoints] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstance = useRef<any>(null);
  const markersLayer = useRef<any>(null);
  const { toast } = useToast();

  // Initialize map
  useEffect(() => {
    if (mapRef.current && !mapInstance.current) {
      // Load Leaflet dynamically
      const loadLeaflet = async () => {
        // Load CSS
        const link = document.createElement('link');
        link.rel = 'stylesheet';
        link.href = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css';
        document.head.appendChild(link);

        // Load JavaScript
        const script = document.createElement('script');
        script.src = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js';
        script.onload = initializeMap;
        document.head.appendChild(script);
      };

      const initializeMap = () => {
        if (window.L && mapRef.current) {
          // Initialize the map centered on Latvia with bounds
          mapInstance.current = window.L.map(mapRef.current, {
            maxBounds: [
              [55.5, 20.5],  // Southwest corner of Latvia
              [58.2, 28.2]   // Northeast corner of Latvia
            ],
            maxBoundsViscosity: 1.0  // Prevent panning outside Latvia
          }).setView([56.8796, 24.6032], 7); // Centered on Latvia with suitable zoom
          
          window.L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          }).addTo(mapInstance.current);

          // Layer group to hold markers, so we can easily clear them
          markersLayer.current = window.L.layerGroup().addTo(mapInstance.current);
        }
      };

      loadLeaflet();
    }

    return () => {
      if (mapInstance.current) {
        mapInstance.current.remove();
        mapInstance.current = null;
      }
    };
  }, []);

  // Update stats when CSV data changes
  useEffect(() => {
    updateStats();
  }, [csvData]);

  const updateStats = () => {
    const lines = csvData.trim().split('\n');
    let validPoints = 0;

    for (let i = 1; i < lines.length; i++) {
      const line = lines[i].trim();
      if (!line) continue;
      
      const parts = line.split(',');
      if (parts.length >= 3) {
        const lon = parseFloat(parts[1].trim());
        const lat = parseFloat(parts[2].trim());
        if (!isNaN(lat) && !isNaN(lon)) {
          validPoints++;
        }
      }
    }

    setTotalPoints(validPoints);
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith('.csv')) {
      toast({
        title: "Invalid file type",
        description: "Please select a CSV file",
        variant: "destructive"
      });
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const content = e.target?.result as string;
      setCsvData(content);
      toast({
        title: "Success!",
        description: "CSV file loaded successfully"
      });
    };
    reader.onerror = () => {
      toast({
        title: "Error",
        description: "Failed to read file",
        variant: "destructive"
      });
    };
    reader.readAsText(file);
  };

  const plotPoints = async () => {
    if (!markersLayer.current || !mapInstance.current) {
      toast({
        title: "Map not ready",
        description: "Please wait for the map to load",
        variant: "destructive"
      });
      return;
    }

    if (!csvData.trim()) {
      toast({
        title: "No data",
        description: "Please provide CSV data first",
        variant: "destructive"
      });
      return;
    }

    setIsLoading(true);
    // Clear previous markers
    markersLayer.current.clearLayers();

    const lines = csvData.trim().split('\n');
    const pointsData: Array<{node: string, lat: number, lon: number}> = [];
    const bounds: Array<[number, number]> = []; // To store coordinates for fitting the map view

    try {
      // Start from 1 to skip header row
      for (let i = 1; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line) continue; // Skip empty lines

        const parts = line.split(',');
        if (parts.length >= 3) {
          const node = parts[0].trim();
          // Assuming X coord is Longitude and Y coord is Latitude
          const lon = parseFloat(parts[1].trim()); // X coord
          const lat = parseFloat(parts[2].trim()); // Y coord

          if (!isNaN(lat) && !isNaN(lon)) {
            pointsData.push({ node, lat, lon });
          } else {
            console.warn(`Skipping invalid coordinates in line: ${line}`);
          }
        } else {
          console.warn(`Skipping malformed line: ${line}`);
        }
      }

      if (pointsData.length === 0) {
        toast({
          title: "No valid coordinates",
          description: "No valid coordinates found in the CSV data",
          variant: "destructive"
        });
        return;
      }

      // Plot all points
      pointsData.forEach(point => {
        const marker = window.L.circleMarker([point.lat, point.lon], {
          radius: 8,
          fillColor: '#10b981',
          color: '#059669',
          weight: 2,
          opacity: 1,
          fillOpacity: 0.8
        }).addTo(markersLayer.current);
        
        marker.bindPopup(`
          <div style="font-family: Inter, sans-serif; min-width: 150px;">
            <div style="background: linear-gradient(135deg, #10b981, #059669); color: white; padding: 8px; margin: -8px -8px 8px -8px; border-radius: 4px;">
              <strong>📍 Node ${point.node}</strong>
            </div>
            <div style="padding: 4px 0;">
              <strong style="color: #10b981;">Latitude:</strong> ${point.lat}<br>
              <strong style="color: #10b981;">Longitude:</strong> ${point.lon}
            </div>
          </div>
        `);
        bounds.push([point.lat, point.lon]);
      });

      // Fit map to show all plotted markers
      if (bounds.length > 0) {
        mapInstance.current.fitBounds(bounds);
        setPlottedPoints(pointsData.length);
        
        toast({
          title: "Success!",
          description: `Successfully plotted all ${pointsData.length} points!`
        });
      }
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to plot points. Please check your CSV format.",
        variant: "destructive"
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-emerald-100 via-teal-50 to-green-100">
      {/* Animated background elements */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute -inset-10 opacity-30">
          <div className="absolute top-1/4 left-1/4 w-72 h-72 bg-emerald-300 rounded-full mix-blend-multiply filter blur-xl animate-pulse"></div>
          <div className="absolute top-1/3 right-1/4 w-72 h-72 bg-teal-300 rounded-full mix-blend-multiply filter blur-xl animate-pulse animation-delay-2000"></div>
          <div className="absolute bottom-1/4 left-1/3 w-72 h-72 bg-green-300 rounded-full mix-blend-multiply filter blur-xl animate-pulse animation-delay-4000"></div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="relative z-10 bg-white/10 backdrop-blur-md border-b border-emerald-200/50 p-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <Button 
              onClick={onBack}
              variant="ghost" 
              className="text-emerald-800 hover:bg-emerald-100/50"
            >
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back to Home
            </Button>
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-gradient-to-br from-emerald-500 to-teal-600 rounded-lg flex items-center justify-center">
                <MapPin className="w-5 h-5 text-white" />
              </div>
              <span className="text-emerald-800 font-semibold text-lg">Roundabout Detector - Latvia Focus</span>
            </div>
          </div>
          
          {/* <div className="flex items-center space-x-2">
            <Button variant="ghost" className="text-emerald-700 hover:bg-emerald-100/50">
              <Settings className="w-4 h-4 mr-2" />
              Settings
            </Button>
          </div> */}
        </div>
      </nav>

      <div className="relative z-10 max-w-7xl mx-auto p-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-[calc(100vh-120px)]">
          {/* Controls Panel */}
          <div className="lg:col-span-1 space-y-6">
            {/* File Upload */}
            <Card className="bg-white/70 backdrop-blur-md border-emerald-200 hover:bg-white/80 transition-all duration-300">
              <CardHeader>
                <CardTitle className="text-emerald-800 flex items-center">
                  <Upload className="w-5 h-5 mr-2" />
                  Upload CSV File
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="relative">
                  <input
                    type="file"
                    accept=".csv"
                    onChange={handleFileUpload}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                  />
                  <div className="border-2 border-dashed border-emerald-300 rounded-lg p-8 text-center hover:border-emerald-400 hover:bg-emerald-50/50 transition-colors">
                    <Upload className="w-8 h-8 text-emerald-500 mx-auto mb-2" />
                    <p className="text-emerald-700 font-medium">Drop CSV file here</p>
                    <p className="text-emerald-600 text-sm">or click to browse</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* CSV Data */}
            <Card className="bg-white/70 backdrop-blur-md border-emerald-200 flex-1">
              <CardHeader>
                <CardTitle className="text-emerald-800">CSV Data</CardTitle>
              </CardHeader>
              <CardContent className="flex-1">
                <Textarea
                  value={csvData}
                  onChange={(e) => setCsvData(e.target.value)}
                  placeholder="Node,X coord,Y coord&#10;1,24.02475,57.06223&#10;2,24.00883,57.07081&#10;..."
                  className="min-h-[200px] bg-emerald-50/50 border-emerald-200 text-emerald-800 placeholder:text-emerald-500 font-mono text-sm focus:border-emerald-400 focus:ring-emerald-200"
                />
              </CardContent>
            </Card>

            {/* Actions & Stats */}
            <Card className="bg-white/70 backdrop-blur-md border-emerald-200">
              <CardContent className="p-6">
                <Button 
                  onClick={plotPoints}
                  disabled={isLoading}
                  className="w-full bg-gradient-to-r from-emerald-500 to-teal-600 hover:from-emerald-600 hover:to-teal-700 text-white font-semibold py-3 mb-4 rounded-xl shadow-lg hover:shadow-emerald-500/25 transition-all duration-300"
                >
                  {isLoading ? (
                    <div className="flex items-center">
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                      Processing...
                    </div>
                  ) : (
                    <>
                      <Eye className="w-4 h-4 mr-2" />
                      Plot All Points
                    </>
                  )}
                </Button>

                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-emerald-50/80 rounded-lg p-4 text-center border border-emerald-200">
                    <div className="text-2xl font-bold text-emerald-600">{totalPoints}</div>
                    <div className="text-emerald-500 text-sm">Total Points</div>
                  </div>
                  <div className="bg-teal-50/80 rounded-lg p-4 text-center border border-teal-200">
                    <div className="text-2xl font-bold text-teal-600">{plottedPoints}</div>
                    <div className="text-teal-500 text-sm">Plotted</div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Map */}
          <div className="lg:col-span-2">
            <Card className="bg-white/70 backdrop-blur-md border-emerald-200 h-full">
              <CardHeader>
                <CardTitle className="text-emerald-800 flex items-center justify-between">
                  <div className="flex items-center">
                    <BarChart3 className="w-5 h-5 mr-2" />
                    Interactive Map - Latvia
                  </div>
                  <Button variant="ghost" size="sm" className="text-emerald-700 hover:bg-emerald-100/50">
                    <Download className="w-4 h-4 mr-2" />
                    Export
                  </Button>
                </CardTitle>
              </CardHeader>
              <CardContent className="p-0 h-[calc(100%-80px)]">
                <div 
                  ref={mapRef} 
                  className="w-full h-full rounded-b-lg border border-emerald-200"
                  style={{ minHeight: '400px' }}
                />
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CSVMapPlotter;
