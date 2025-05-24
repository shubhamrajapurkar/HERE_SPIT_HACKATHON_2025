
import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { MapPin, Upload, Eye, Zap, Globe, BarChart3, ArrowRight, Star } from 'lucide-react';
import CSVMapPlotter from '@/components/CSVMapPlotter';

const Index = () => {
  const [showPlotter, setShowPlotter] = useState(false);

  if (showPlotter) {
    return <CSVMapPlotter onBack={() => setShowPlotter(false)} />;
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-emerald-100 via-teal-50 to-green-100 overflow-hidden">
      {/* Animated background elements */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute -inset-10 opacity-30">
          <div className="absolute top-1/4 left-1/4 w-72 h-72 bg-emerald-300 rounded-full mix-blend-multiply filter blur-xl animate-pulse"></div>
          <div className="absolute top-1/3 right-1/4 w-72 h-72 bg-teal-300 rounded-full mix-blend-multiply filter blur-xl animate-pulse animation-delay-2000"></div>
          <div className="absolute bottom-1/4 left-1/3 w-72 h-72 bg-green-300 rounded-full mix-blend-multiply filter blur-xl animate-pulse animation-delay-4000"></div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="relative z-10 p-6">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="w-10 h-10 bg-gradient-to-br from-emerald-500 to-teal-600 rounded-xl flex items-center justify-center">
              <MapPin className="w-6 h-6 text-white" />
            </div>
            <span className="text-emerald-800 font-bold text-xl">Roundabout Detector</span>
          </div>
          {/* <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-1">
              {[...Array(5)].map((_, i) => (
                <Star key={i} className="w-4 h-4 text-amber-400 fill-current" />
              ))}
              <span className="text-emerald-700 ml-2">4.9/5</span>
            </div>
          </div> */}
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative z-10 px-6 pt-20 pb-32">
        <div className="max-w-7xl mx-auto text-center">
          <div className="animate-fade-in">
            <div className="inline-flex items-center px-4 py-2 bg-emerald-100/80 backdrop-blur-md rounded-full text-emerald-800 text-sm font-medium mb-8 border border-emerald-200">
              <Zap className="w-4 h-4 mr-2 text-emerald-600" />
              Fast • Secure • Beautiful
            </div>
            
            <h1 className="text-5xl md:text-7xl font-bold text-emerald-900 mb-6 leading-tight">
              Angular-based cycle deletection
              <span className="block bg-gradient-to-r from-emerald-600 to-teal-600 bg-clip-text text-transparent pb-10">
                with indegree filtering
              </span>
            </h1>
            
            <p className="text-xl md:text-2xl text-emerald-700 mb-12 max-w-3xl mx-auto leading-relaxed">
              Transform your CSV coordinates into stunning interactive maps with clustering, 
              real-time visualization, and beautiful markers.
            </p>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-16">
              <Button 
                onClick={() => setShowPlotter(true)}
                className="bg-gradient-to-r from-emerald-500 to-teal-600 hover:from-emerald-600 hover:to-teal-700 text-white px-8 py-4 text-lg font-semibold rounded-xl shadow-2xl hover:shadow-emerald-500/25 transition-all duration-300 hover:scale-105 group"
              >
                Start Plotting Now
                <ArrowRight className="w-5 h-5 ml-2 group-hover:translate-x-1 transition-transform" />
              </Button>
              {/* <Button 
                variant="outline" 
                className="border-emerald-300 text-emerald-700 hover:bg-emerald-50 px-8 py-4 text-lg font-semibold rounded-xl backdrop-blur-sm transition-all duration-300"
              >
                View Demo
              </Button> */}
            </div>
          </div>

          {/* Feature Preview Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl mx-auto">
            {[
              { icon: Upload, title: "Easy Upload", desc: "Drag & drop CSV files" },
              { icon: Eye, title: "Real-time Preview", desc: "Instant map visualization" },
              { icon: Globe, title: "Interactive Maps", desc: "Zoom, cluster & explore" }
            ].map((feature, index) => (
              <Card key={index} className="bg-white/80 backdrop-blur-md border-emerald-200 hover:bg-white/90 transition-all duration-300 hover:scale-105 group">
                <CardContent className="p-6 text-center">
                  <div className="w-12 h-12 bg-gradient-to-br from-emerald-500 to-teal-600 rounded-xl flex items-center justify-center mx-auto mb-4 group-hover:scale-110 transition-transform duration-300">
                    <feature.icon className="w-6 h-6 text-white" />
                  </div>
                  <h3 className="text-emerald-800 font-semibold text-lg mb-2">{feature.title}</h3>
                  <p className="text-emerald-600">{feature.desc}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="relative z-10 px-6 py-20">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl md:text-5xl font-bold text-emerald-900 mb-6">
              Powerful Features
            </h2>
            <p className="text-xl text-emerald-700 max-w-2xl mx-auto">
              Everything you need to create stunning geographic visualizations
            </p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
            <div className="space-y-8">
              {[
                {
                  icon: BarChart3,
                  title: "Smart Clustering",
                  description: "Automatically group nearby points for better performance and cleaner visualization"
                },
                {
                  icon: Zap,
                  title: "Lightning Fast",
                  description: "Process thousands of coordinates in seconds with optimized rendering"
                },
                {
                  icon: Globe,
                  title: "Multiple Map Styles",
                  description: "Choose from various map themes and customize markers to match your brand"
                }
              ].map((feature, index) => (
                <div key={index} className="flex items-start space-x-4 group">
                  <div className="w-12 h-12 bg-gradient-to-br from-emerald-500 to-teal-600 rounded-xl flex items-center justify-center flex-shrink-0 group-hover:scale-110 transition-transform duration-300">
                    <feature.icon className="w-6 h-6 text-white" />
                  </div>
                  <div>
                    <h3 className="text-emerald-800 font-semibold text-xl mb-2">{feature.title}</h3>
                    <p className="text-emerald-600 leading-relaxed">{feature.description}</p>
                  </div>
                </div>
              ))}
            </div>

            <div className="relative">
              <div className="bg-white/60 backdrop-blur-md rounded-2xl p-8 border border-emerald-200">
                <div className="w-full h-64 bg-gradient-to-br from-emerald-100 to-teal-100 rounded-xl flex items-center justify-center">
                  <div className="text-center">
                    <MapPin className="w-16 h-16 text-emerald-500 mx-auto mb-4" />
                    <p className="text-emerald-700">Interactive Map Preview</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="relative z-10 px-6 py-20">
        <div className="max-w-4xl mx-auto text-center">
          <Card className="bg-white/70 backdrop-blur-md border-emerald-200 p-12">
            <CardContent className="p-0">
              <h2 className="text-3xl md:text-4xl font-bold text-emerald-900 mb-6">
                Ready to Map Your Data?
              </h2>
              <p className="text-emerald-700 text-lg mb-8 max-w-2xl mx-auto">
                Join thousands of users who trust Roundabout Detector for their geographic data visualization needs.
              </p>
              <Button 
                onClick={() => setShowPlotter(true)}
                className="bg-gradient-to-r from-emerald-500 to-teal-600 hover:from-emerald-600 hover:to-teal-700 text-white px-12 py-4 text-lg font-semibold rounded-xl shadow-2xl hover:shadow-emerald-500/25 transition-all duration-300 hover:scale-105 group"
              >
                Get Started Free
                <ArrowRight className="w-5 h-5 ml-2 group-hover:translate-x-1 transition-transform" />
              </Button>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Footer */}
      <footer className="relative z-10 px-6 py-8 border-t border-emerald-200">
        <div className="max-w-7xl mx-auto text-center">
          <p className="text-emerald-600">
            © 2024 Roundabout Detector. Made with ❤️ by <a href="https://in.linkedin.com/in/sujaldas">Sujal</a>, <a href="https://in.linkedin.com/in/miraj-lakeshri-242693254">Miraj</a>, <a href="https://www.linkedin.com/in/pranay-fadtare-300317283/">Pranay</a> and <a href="https://www.linkedin.com/in/raj-nandurkar-3546b9236">Raj</a>.
          </p>
        </div>
      </footer>
    </div>
  );
};

export default Index;
