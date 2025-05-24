import { useState, useEffect } from 'react';
import Modal from '../components/Modal';
import { Sparkles, Film, Music, Calendar, Gamepad2, ArrowRight } from 'lucide-react';

const entertainmentTabs = [
  {
    key: 'movies',
    label: 'Movies',
    icon: <Film className="w-6 h-6 text-red-500" />,
    content: [
      { title: 'In Theatres', items: ['Dune: Part Two', 'Godzilla x Kong', 'Kung Fu Panda 4'] },
      { title: 'Trending', items: ['Oppenheimer', 'Barbie', 'Jawan'] },
    ],
  },
  {
    key: 'music',
    label: 'Music',
    icon: <Music className="w-6 h-6 text-blue-500" />,
    content: [
      { title: 'Top Charts', items: ['Calm Down - Rema', 'Flowers - Miley Cyrus', 'Unholy - Sam Smith'] },
      { title: 'New Releases', items: ['Endless Summer', 'Midnight Memories', 'Golden Hour'] },
    ],
  },
  {
    key: 'events',
    label: 'Events',
    icon: <Calendar className="w-6 h-6 text-green-500" />,
    content: [
      { title: 'Nearby', items: ['Food Festival', 'Live Concert', 'Art Expo'] },
      { title: 'Online', items: ['Webinar: AI in 2024', 'Virtual Standup Comedy'] },
    ],
  },
  {
    key: 'games',
    label: 'Games',
    icon: <Gamepad2 className="w-6 h-6 text-purple-500" />,
    content: [
      { title: 'Popular', items: ['Valorant', 'FIFA 24', 'Among Us'] },
      { title: 'New Releases', items: ['Starfield', 'Hades II'] },
    ],
  },
];

// eslint-disable-next-line react/prop-types
function EntertainmentPage({ isOpen, onClose }) { 
  const [activeTab, setActiveTab] = useState(entertainmentTabs[0].key);
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    if (isOpen) {
      setTimeout(() => setIsLoaded(true), 100);
    } else {
      setIsLoaded(false);
    }
  }, [isOpen]);

  const currentTab = entertainmentTabs.find(tab => tab.key === activeTab);

  return (
    <Modal isOpen={isOpen} onClose={onClose}>
      <div className="min-h-[60vh] bg-gradient-to-br from-blue-50 via-pink-50 to-purple-50 relative overflow-hidden rounded-2xl">
        {/* Animated background */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute -top-32 -right-32 w-64 h-64 bg-gradient-to-br from-blue-200/30 to-pink-200/30 rounded-full blur-3xl animate-pulse"></div>
          <div className="absolute -bottom-32 -left-32 w-64 h-64 bg-gradient-to-br from-purple-200/30 to-pink-200/30 rounded-full blur-3xl animate-pulse" style={{animationDelay: '1s'}}></div>
        </div>

        {/* Header */}
        <div className={`bg-white/80 shadow-sm border-b border-gray-100 transition-all duration-1000 rounded-t-2xl ${isLoaded ? 'translate-y-0 opacity-100' : '-translate-y-10 opacity-0'}`}>
          <div className="max-w-3xl mx-auto px-6 py-6 text-center">
            <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full mb-4 relative overflow-hidden group cursor-pointer">
              <Sparkles className="w-8 h-8 text-white transition-transform duration-500 group-hover:scale-110" />
            </div>
            <h1 className="text-3xl font-bold text-gray-900 mb-2 bg-gradient-to-r from-blue-900 via-purple-800 to-pink-700 bg-clip-text text-transparent">
              Explore Entertainment
            </h1>
            <p className="text-gray-600 text-lg max-w-xl mx-auto leading-relaxed">
              Discover movies, music, events, and games happening around you or online!
            </p>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex justify-center gap-4 mt-4">
          {entertainmentTabs.map(tab => (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
              className={`flex flex-col items-center px-4 py-2 rounded-xl transition-all duration-300 font-medium text-sm focus:outline-none ${
                activeTab === tab.key
                  ? 'bg-gradient-to-r from-blue-100 to-purple-100 text-blue-700 shadow-md scale-105'
                  : 'bg-white text-gray-500 hover:bg-blue-50 hover:text-blue-600'
              }`}
            >
              {tab.icon}
              <span className="mt-1">{tab.label}</span>
            </button>
          ))}
        </div>

        {/* Tab Content */}
        <div className="max-w-3xl mx-auto px-6 py-8">
          {currentTab.content.map((section, idx) => (
            <div key={section.title} className={`mb-8 transition-all duration-700 ${isLoaded ? 'translate-y-0 opacity-100' : 'translate-y-10 opacity-0'}`} style={{ transitionDelay: `${idx * 150}ms` }}>
              <div className="flex items-center gap-3 mb-2">
                <ArrowRight className="w-4 h-4 text-blue-400" />
                <h2 className="text-xl font-semibold text-gray-800">{section.title}</h2>
              </div>
              <div className="flex flex-wrap gap-4 mt-2">
                {section.items.map(item => (
                  <div key={item} className="px-5 py-3 bg-white rounded-2xl shadow hover:shadow-lg border border-gray-100 text-gray-700 font-medium text-base transition-all duration-300 cursor-pointer hover:bg-gradient-to-r hover:from-blue-50 hover:to-pink-50">
                    {item}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </Modal>
  );
}

export default EntertainmentPage; 