import React, { useState } from 'react';
import { 
  Database, 
  Cpu, 
  RefreshCw, 
  Shield,
  Plus,
  Filter,
  MapPin,
  Layers,
  Navigation,
  Camera,
  Truck,
  Satellite,
  BarChart3,
  Settings,
  Eye,
  Zap,
  Activity
} from 'lucide-react';

const Dashboard = () => {
  const [activeFilter, setActiveFilter] = useState('all');

  const dataLayers = [
    {
      id: 1,
      name: 'Open Street Maps',
      type: 'Vector Tiles',
      icon: MapPin,
      status: 'Active',
      statusColor: 'bg-green-500',
      bgColor: 'bg-blue-50',
      iconColor: 'bg-blue-500',
      lastUpdate: '2 min ago',
      coverage: '99.8%',
      quality: 95,
      gradientFrom: 'from-green-400',
      gradientTo: 'to-blue-500'
    },
    {
      id: 2,
      name: 'JustDial Data',
      type: 'POI Data',
      icon: MapPin,
      status: 'Inactive',
      statusColor: 'bg-red-500',
      bgColor: 'bg-purple-50',
      iconColor: 'bg-purple-500',
      lastUpdate: '2 hours ago',
      coverage: '92.1%',
      quality: 95,
      gradientFrom: 'from-green-400',
      gradientTo: 'to-blue-500'
    },
    {
      id: 3,
      name: 'Wikipedia',
      type: 'Navigation',
      icon: Navigation,
      status: 'Active',
      statusColor: 'bg-green-500',
      bgColor: 'bg-emerald-50',
      iconColor: 'bg-emerald-500',
      lastUpdate: '10 min ago',
      coverage: '98.9%',
      quality: 95,
      gradientFrom: 'from-green-400',
      gradientTo: 'to-blue-500'
    },
    {
      id: 4,
      name: 'HERE Satellite',
      type: 'Imagery',
      icon: Satellite,
      status: 'Active',
      statusColor: 'bg-green-500',
      bgColor: 'bg-green-50',
      iconColor: 'bg-green-500',
      lastUpdate: '5 min ago',
      coverage: '97.2%',
      quality: 98,
      gradientFrom: 'from-green-400',
      gradientTo: 'to-blue-500'
    },
    {
      id: 5,
      name: 'Traffic Cameras',
      type: 'Real-time Video',
      icon: Camera,
      status: 'Active',
      statusColor: 'bg-green-500',
      bgColor: 'bg-orange-50',
      iconColor: 'bg-orange-500',
      lastUpdate: '1 min ago',
      coverage: '94.5%',
      quality: 92,
      gradientFrom: 'from-green-400',
      gradientTo: 'to-blue-500'
    },
    {
      id: 6,
      name: 'Fleet Tracking',
      type: 'Vehicle IoT',
      icon: Truck,
      status: 'Active',
      statusColor: 'bg-green-500',
      bgColor: 'bg-cyan-50',
      iconColor: 'bg-cyan-500',
      lastUpdate: '3 min ago',
      coverage: '96.7%',
      quality: 97,
      gradientFrom: 'from-green-400',
      gradientTo: 'to-blue-500'
    }
  ];

  const stats = [
    { title: 'Active Data Sources', value: '24', icon: Database, color: 'from-blue-500 to-blue-600', textColor: 'text-blue-600' },
    { title: 'Processing Jobs', value: '12', icon: Activity, color: 'from-purple-500 to-purple-600', textColor: 'text-purple-600' },
    { title: 'Map Updates Today', value: '156', icon: RefreshCw, color: 'from-emerald-500 to-emerald-600', textColor: 'text-emerald-600' },
    { title: 'Quality Score', value: '94%', icon: Shield, color: 'from-orange-500 to-orange-600', textColor: 'text-orange-600' }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-blue-50 to-indigo-50">
      {/* Header */}
      <div className="bg-white/80 backdrop-blur-sm border-b border-gray-200/50 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-indigo-600 rounded-full flex items-center justify-center">
                <MapPin className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-gray-900 to-gray-700 bg-clip-text text-transparent">
                  HERE <span className="text-blue-600">Location Intelligence</span>
                </h1>
                <p className="text-sm text-gray-600">Powering location-aware applications with real-time geospatial data</p>
              </div>
            </div>
            <div className="flex items-center space-x-3">
              <button className="flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-xl hover:from-blue-600 hover:to-blue-700 transition-all duration-200 shadow-lg shadow-blue-500/25">
                <RefreshCw className="h-4 w-4" />
                <span className="font-medium">Refresh</span>
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-8 space-y-8">
        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {stats.map((stat, index) => (
            <div key={index} className="group relative overflow-hidden bg-white/70 backdrop-blur-sm rounded-2xl shadow-lg shadow-gray-200/50 border border-white/50 p-6 hover:shadow-xl hover:shadow-gray-200/60 transition-all duration-300">
              <div className="absolute inset-0 bg-gradient-to-br opacity-5 group-hover:opacity-10 transition-opacity duration-300" 
                   style={{ backgroundImage: `linear-gradient(135deg, ${stat.color.split(' ')[1]}, ${stat.color.split(' ')[3]})` }}></div>
              <div className="relative flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600 mb-1">{stat.title}</p>
                  <p className="text-3xl font-bold text-gray-900">{stat.value}</p>
                </div>
                <div className={`p-3 rounded-xl bg-gradient-to-r ${stat.color} shadow-lg`}>
                  <stat.icon className="h-6 w-6 text-white" />
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Data Layer Management */}
        <div className="bg-white/70 backdrop-blur-sm rounded-2xl shadow-xl shadow-gray-200/50 border border-white/50 overflow-hidden">
          <div className="p-6 border-b border-gray-200/50 bg-gradient-to-r from-white/80 to-gray-50/80">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-gradient-to-r from-blue-500 to-indigo-600 rounded-xl">
                  <Layers className="h-5 w-5 text-white" />
                </div>
                <h2 className="text-xl font-bold text-gray-900">Data Layer Management</h2>
              </div>
              <div className="flex space-x-3">
                <button className="flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-xl hover:from-blue-600 hover:to-blue-700 transition-all duration-200 shadow-lg shadow-blue-500/25">
                  <Plus className="h-4 w-4" />
                  <span className="font-medium">Add Data Source</span>
                </button>
                <button className="flex items-center space-x-2 px-4 py-2 bg-white/80 text-gray-700 rounded-xl border border-gray-200 hover:bg-white hover:shadow-md transition-all duration-200">
                  <Filter className="h-4 w-4" />
                  <span className="font-medium">Filter Layers</span>
                </button>
              </div>
            </div>
          </div>

          <div className="p-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
              {dataLayers.map((layer) => (
                <div key={layer.id} className={`group relative overflow-hidden ${layer.bgColor} rounded-2xl border border-white/50 p-6 hover:shadow-xl transition-all duration-300 hover:-translate-y-1`}>
                  <div className="absolute top-0 right-0 w-20 h-20 bg-gradient-to-br from-white/20 to-transparent rounded-bl-full"></div>
                  
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center space-x-3">
                      <div className={`p-3 ${layer.iconColor} rounded-xl shadow-lg`}>
                        <layer.icon className="h-6 w-6 text-white" />
                      </div>
                      <div>
                        <h3 className="font-bold text-gray-900 text-lg">{layer.name}</h3>
                        <p className="text-sm text-gray-600">{layer.type}</p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <div className={`w-2 h-2 ${layer.statusColor} rounded-full`}></div>
                      <span className={`text-xs font-medium px-2 py-1 rounded-full ${
                        layer.status === 'Active' 
                          ? 'bg-green-100 text-green-700' 
                          : 'bg-red-100 text-red-700'
                      }`}>
                        {layer.status}
                      </span>
                    </div>
                  </div>

                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <p className="text-xs text-gray-500 mb-1">Last Update</p>
                        <p className="text-sm font-medium text-gray-900">{layer.lastUpdate}</p>
                      </div>
                      <div>
                        <p className="text-xs text-gray-500 mb-1">Coverage</p>
                        <div className="flex items-center space-x-1">
                          <MapPin className="h-3 w-3 text-blue-500" />
                          <p className="text-sm font-medium text-gray-900">{layer.coverage}</p>
                        </div>
                      </div>
                    </div>

                    <div>
                      <div className="flex justify-between items-center mb-2">
                        <p className="text-xs text-gray-500">Data Quality</p>
                        <div className="flex items-center space-x-1">
                          <Zap className="h-3 w-3 text-green-500" />
                          <span className="text-sm font-medium text-gray-900">{layer.quality}%</span>
                        </div>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
                        <div 
                          className={`h-2 bg-gradient-to-r ${layer.gradientFrom} ${layer.gradientTo} rounded-full transition-all duration-500`}
                          style={{ width: `${layer.quality}%` }}
                        ></div>
                      </div>
                    </div>

                    <div className="flex space-x-2 pt-2">
                      <button className="flex-1 flex items-center justify-center space-x-2 px-3 py-2 bg-blue-500 text-white rounded-xl hover:bg-blue-600 transition-colors duration-200 text-sm font-medium">
                        <Settings className="h-4 w-4" />
                        <span>Configure</span>
                      </button>
                      <button className="flex-1 flex items-center justify-center space-x-2 px-3 py-2 bg-white/80 text-gray-700 rounded-xl border border-gray-200 hover:bg-white hover:shadow-md transition-all duration-200 text-sm font-medium">
                        <Eye className="h-4 w-4" />
                        <span>View Data</span>
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Processing Pipeline */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white/70 backdrop-blur-sm rounded-2xl shadow-xl shadow-gray-200/50 border border-white/50 p-6">
            <div className="flex items-center space-x-3 mb-6">
              <div className="p-2 bg-gradient-to-r from-purple-500 to-purple-600 rounded-xl">
                <Activity className="h-5 w-5 text-white" />
              </div>
              <h3 className="text-lg font-bold text-gray-900">Processing Pipeline</h3>
            </div>
            <div className="space-y-4">
              {[
                { stage: 'Data Ingestion', progress: 100, status: 'complete', color: 'from-green-400 to-green-500' },
                { stage: 'Feature Detection', progress: 75, status: 'active', color: 'from-blue-400 to-blue-500' },
                { stage: 'Semantic Enrichment', progress: 45, status: 'active', color: 'from-purple-400 to-purple-500' },
                { stage: 'Quality Check', progress: 0, status: 'pending', color: 'from-gray-300 to-gray-400' }
              ].map((stage, index) => (
                <div key={index} className="space-y-3 p-4 bg-gradient-to-r from-gray-50/50 to-white/50 rounded-xl border border-white/50">
                  <div className="flex justify-between items-center">
                    <div className="flex items-center space-x-3">
                      <div className={`w-3 h-3 rounded-full ${
                        stage.status === 'complete' ? 'bg-green-500' : 
                        stage.status === 'active' ? 'bg-blue-500 animate-pulse' : 'bg-gray-300'
                      }`}></div>
                      <span className="font-medium text-gray-900">{stage.stage}</span>
                    </div>
                    <span className="text-sm font-medium text-gray-600">{stage.progress}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
                    <div 
                      className={`h-2 bg-gradient-to-r ${stage.color} rounded-full transition-all duration-700 ease-out`}
                      style={{ width: `${stage.progress}%` }}
                    ></div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-white/70 backdrop-blur-sm rounded-2xl shadow-xl shadow-gray-200/50 border border-white/50 p-6">
            <div className="flex items-center space-x-3 mb-6">
              <div className="p-2 bg-gradient-to-r from-emerald-500 to-emerald-600 rounded-xl">
                <BarChart3 className="h-5 w-5 text-white" />
              </div>
              <h3 className="text-lg font-bold text-gray-900">Recent Activity</h3>
            </div>
            <div className="space-y-4">
              {[
                { action: 'Satellite imagery processed', time: '2 min ago', status: 'success', icon: Satellite },
                { action: 'Feature extraction completed', time: '5 min ago', status: 'success', icon: Cpu },
                { action: 'Quality check failed', time: '12 min ago', status: 'error', icon: Shield },
                { action: 'New GPS traces ingested', time: '18 min ago', status: 'success', icon: Navigation }
              ].map((activity, index) => (
                <div key={index} className="flex items-center space-x-4 p-3 bg-gradient-to-r from-gray-50/50 to-white/50 rounded-xl border border-white/50 hover:shadow-md transition-all duration-200">
                  <div className={`p-2 rounded-lg ${
                    activity.status === 'success' ? 'bg-green-100 text-green-600' : 'bg-red-100 text-red-600'
                  }`}>
                    <activity.icon className="h-4 w-4" />
                  </div>
                  <div className="flex-1">
                    <p className="font-medium text-gray-900">{activity.action}</p>
                    <p className="text-xs text-gray-500">{activity.time}</p>
                  </div>
                  <div className={`w-2 h-2 rounded-full ${
                    activity.status === 'success' ? 'bg-green-500' : 'bg-red-500'
                  }`}></div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;