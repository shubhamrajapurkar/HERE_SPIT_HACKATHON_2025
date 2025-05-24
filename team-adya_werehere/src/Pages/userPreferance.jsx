import React, { useState, useEffect } from 'react';
import {
  ChefHat, Heart, AlertTriangle, Globe, Flame, Check, Star, Sparkles, ArrowRight
} from 'lucide-react';

const dietaryOptions = {
  diet_type: [
    { value: "vegetarian", label: "Vegetarian", icon: "🥗", color: "bg-green-50 border-green-200 text-green-700" },
    { value: "vegan", label: "Vegan", icon: "🌱", color: "bg-emerald-50 border-emerald-200 text-emerald-700" },
    { value: "pescatarian", label: "Pescatarian", icon: "🐟", color: "bg-blue-50 border-blue-200 text-blue-700" },
    { value: "gluten-free", label: "Gluten-Free", icon: "🌾", color: "bg-amber-50 border-amber-200 text-amber-700" },
    { value: "keto", label: "Keto", icon: "🥑", color: "bg-lime-50 border-lime-200 text-lime-700" },
    { value: "paleo", label: "Paleo", icon: "🥩", color: "bg-orange-50 border-orange-200 text-orange-700" },
    { value: "halal", label: "Halal", icon: "☪️", color: "bg-teal-50 border-teal-200 text-teal-700" },
    { value: "kosher", label: "Kosher", icon: "✡️", color: "bg-indigo-50 border-indigo-200 text-indigo-700" },
    { value: "lactose-free", label: "Lactose-Free", icon: "🥛", color: "bg-purple-50 border-purple-200 text-purple-700" }
  ],
  allergies: [
    { value: "peanuts", label: "Peanuts", icon: "🥜", color: "bg-red-50 border-red-200 text-red-700" },
    { value: "tree nuts", label: "Tree Nuts", icon: "🌰", color: "bg-red-50 border-red-200 text-red-700" },
    { value: "dairy", label: "Dairy", icon: "🧀", color: "bg-red-50 border-red-200 text-red-700" },
    { value: "eggs", label: "Eggs", icon: "🥚", color: "bg-red-50 border-red-200 text-red-700" },
    { value: "shellfish", label: "Shellfish", icon: "🦐", color: "bg-red-50 border-red-200 text-red-700" },
    { value: "soy", label: "Soy", icon: "🫘", color: "bg-red-50 border-red-200 text-red-700" },
    { value: "gluten", label: "Gluten", icon: "🍞", color: "bg-red-50 border-red-200 text-red-700" },
    { value: "sesame", label: "Sesame", icon: "🌾", color: "bg-red-50 border-red-200 text-red-700" }
  ],
  disliked_ingredients: [
    { value: "cilantro", label: "Cilantro", icon: "🌿", color: "bg-gray-50 border-gray-200 text-gray-700" },
    { value: "onion", label: "Onion", icon: "🧅", color: "bg-gray-50 border-gray-200 text-gray-700" },
    { value: "mushroom", label: "Mushroom", icon: "🍄", color: "bg-gray-50 border-gray-200 text-gray-700" }
  ],
  preferred_cuisines: [
    { value: "italian", label: "Italian", icon: "🍝", color: "bg-rose-50 border-rose-200 text-rose-700" },
    { value: "indian", label: "Indian", icon: "🍛", color: "bg-orange-50 border-orange-200 text-orange-700" },
    { value: "thai", label: "Thai", icon: "🍜", color: "bg-green-50 border-green-200 text-green-700" },
    { value: "japanese", label: "Japanese", icon: "🍣", color: "bg-pink-50 border-pink-200 text-pink-700" },
    { value: "mexican", label: "Mexican", icon: "🌮", color: "bg-yellow-50 border-yellow-200 text-yellow-700" }
  ],
  spice_tolerance: [
    { value: "low", label: "Mild", icon: "🌶️", color: "bg-green-100 border-green-300" },
    { value: "medium", label: "Medium", icon: "🌶️🌶️", color: "bg-yellow-100 border-yellow-300" },
    { value: "high", label: "Spicy", icon: "🌶️🌶️🌶️", color: "bg-red-100 border-red-300" }
  ]
};

const categoryIcons = {
  diet_type: ChefHat,
  allergies: AlertTriangle,
  disliked_ingredients: Heart,
  preferred_cuisines: Globe,
  spice_tolerance: Flame
};

const categoryTitles = {
  diet_type: "Diet Type",
  allergies: "Allergies & Restrictions",
  disliked_ingredients: "Disliked Ingredients",
  preferred_cuisines: "Favorite Cuisines",
  spice_tolerance: "Spice Tolerance"
};

export default function UserPreferencePage({ onClose }) {
  const [isLoaded, setIsLoaded] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [completedSections, setCompletedSections] = useState(new Set());

  const [preferences, setPreferences] = useState({
    diet_type: [],
    allergies: [],
    disliked_ingredients: [],
    preferred_cuisines: [],
    spice_tolerance: ""
  });

  useEffect(() => {
    setTimeout(() => setIsLoaded(true), 100);
  }, []);

  useEffect(() => {
    // Track completed sections for progress indication
    const completed = new Set();
    Object.entries(preferences).forEach(([key, value]) => {
      if (Array.isArray(value) ? value.length > 0 : value !== "") {
        completed.add(key);
      }
    });
    setCompletedSections(completed);
  }, [preferences]);

  const handleCheckboxChange = (category, value) => {
    setPreferences(prev => {
      const isSelected = prev[category].includes(value);
      return {
        ...prev,
        [category]: isSelected
          ? prev[category].filter(item => item !== value)
          : [...prev[category], value]
      };
    });
  };

  const handleSpiceChange = (value) => {
    setPreferences(prev => ({ ...prev, spice_tolerance: value }));
  };

  const handleSubmit = async () => {
    setIsSubmitting(true);

    try {
      localStorage.setItem('userPreferences', JSON.stringify(preferences));

      await new Promise(resolve => setTimeout(resolve, 1000));

      if (onClose) {
        onClose();
      }
    } catch (error) {
      console.error('Error saving preferences:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  const getSelectionCount = () => {
    return Object.values(preferences).reduce((count, value) => {
      if (Array.isArray(value)) return count + value.length;
      return value ? count + 1 : count;
    }, 0);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-orange-50 via-red-50 to-pink-50 relative overflow-hidden">
      {/* Animated background elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-gradient-to-br from-orange-200/30 to-red-200/30 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-gradient-to-br from-pink-200/30 to-purple-200/30 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }}></div>
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-gradient-to-br from-yellow-200/20 to-orange-200/20 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '2s' }}></div>
      </div>

      {/* Progress indicator */}
      <div className="sticky top-0 z-50 bg-white/80 backdrop-blur-lg border-b border-white/20 shadow-sm">
        <div className="max-w-4xl mx-auto px-6 py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2">
                <Sparkles className="w-5 h-5 text-orange-500" />
                <span className="text-sm font-medium text-gray-700">
                  {getSelectionCount()} preferences selected
                </span>
              </div>
            </div>
            <div className="flex items-center gap-2">
              {Object.keys(dietaryOptions).map((category, index) => (
                <div
                  key={category}
                  className={`w-2 h-2 rounded-full transition-all duration-500 ${completedSections.has(category)
                    ? 'bg-gradient-to-r from-orange-500 to-red-500 scale-125'
                    : 'bg-gray-200'
                    }`}
                  style={{ transitionDelay: `${index * 100}ms` }}
                />
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Header */}
      <div className={`bg-white shadow-sm border-b border-gray-100 transition-all duration-1000 ${isLoaded ? 'translate-y-0 opacity-100' : '-translate-y-10 opacity-0'
        }`}>
        <div className="max-w-4xl mx-auto px-6 py-8 text-center">
          <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-r from-orange-500 to-red-500 rounded-full mb-6 relative overflow-hidden group cursor-pointer">
            <ChefHat className="w-10 h-10 text-white transition-transform duration-500 group-hover:scale-110" />
            <div className="absolute inset-0 bg-gradient-to-r from-red-500 to-pink-500 opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
          </div>
          <h1 className="text-4xl font-bold text-gray-900 mb-3 bg-gradient-to-r from-gray-900 via-gray-800 to-gray-900 bg-clip-text">
            Customize Your Food Preferences
          </h1>
          <p className="text-gray-600 text-xl max-w-2xl mx-auto leading-relaxed">
            Tell us what you love, and we'll personalize your dining experience with perfect recommendations
          </p>
        </div>
      </div>

      {/* Main content */}
      <div className="max-w-4xl mx-auto px-6 py-8 space-y-8">
        {Object.entries(dietaryOptions).map(([category, options], categoryIndex) => {
          const IconComponent = categoryIcons[category];
          const isCompleted = completedSections.has(category);

          return (
            <div
              key={category}
              className={`transition-all duration-700 ${isLoaded ? 'translate-y-0 opacity-100' : 'translate-y-10 opacity-0'
                }`}
              style={{ transitionDelay: `${categoryIndex * 200}ms` }}
            >
              <div className={`bg-white rounded-3xl shadow-xl border border-gray-100 overflow-hidden hover:shadow-2xl transition-all duration-500 hover:-translate-y-1 ${isCompleted ? 'ring-2 ring-orange-200 shadow-orange-100/50' : ''
                }`}>
                <div className="bg-gradient-to-r from-gray-50 to-gray-100 px-6 py-5 border-b border-gray-200 relative overflow-hidden">
                  <div className="flex items-center gap-4 relative z-10">
                    <div className={`p-3 bg-white rounded-xl shadow-md transition-all duration-500 ${isCompleted ? 'bg-gradient-to-r from-orange-50 to-red-50 shadow-orange-200/50' : ''
                      }`}>
                      <IconComponent className={`w-6 h-6 transition-colors duration-500 ${isCompleted ? 'text-orange-600' : 'text-gray-700'
                        }`} />
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center gap-3">
                        <h3 className="text-2xl font-bold text-gray-900">{categoryTitles[category]}</h3>
                        {isCompleted && (
                          <div className="animate-bounce">
                            <Check className="w-5 h-5 text-green-500" />
                          </div>
                        )}
                      </div>
                      <p className="text-sm text-gray-600 mt-1">
                        {category === 'diet_type' && 'Select your dietary preferences to find perfect matches'}
                        {category === 'allergies' && 'Help us keep you safe with allergy information'}
                        {category === 'disliked_ingredients' && 'What would you prefer to avoid in your meals?'}
                        {category === 'preferred_cuisines' && 'Your favorite food styles from around the world'}
                        {category === 'spice_tolerance' && 'How much heat can you handle?'}
                      </p>
                    </div>
                  </div>
                  {isCompleted && (
                    <div className="absolute inset-0 bg-gradient-to-r from-orange-500/5 to-red-500/5 animate-pulse"></div>
                  )}
                </div>

                <div className="p-8">
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                    {category === "spice_tolerance" ? (
                      options.map((option, index) => (
                        <label key={option.value} className="cursor-pointer group">
                          <input
                            type="radio"
                            name={category}
                            value={option.value}
                            checked={preferences.spice_tolerance === option.value}
                            onChange={() => handleSpiceChange(option.value)}
                            className="hidden"
                          />
                          <div
                            className={`flex items-center gap-3 px-5 py-4 rounded-2xl border-2 cursor-pointer transition-all duration-300 group-hover:scale-105 ${preferences.spice_tolerance === option.value
                              ? 'border-red-400 bg-gradient-to-r from-red-50 to-orange-50 text-red-700 shadow-lg shadow-red-100'
                              : "border-gray-200 bg-white text-gray-700 hover:bg-gray-50 hover:border-gray-300 hover:shadow-md"
                              }`}
                            style={{ transitionDelay: `${index * 50}ms` }}
                          >
                            <span className="text-3xl animate-pulse">{option.icon}</span>
                            <span className="font-bold text-lg">{option.label}</span>
                          </div>
                        </label>
                      ))
                    ) : (
                      options.map((option, index) => {
                        const isChecked = preferences[category].includes(option.value);
                        return (
                          <label
                            key={option.value}
                            className="cursor-pointer group"
                            style={{ transitionDelay: `${index * 50}ms` }}
                          >
                            <input
                              type="checkbox"
                              value={option.value}
                              checked={isChecked}
                              onChange={() => handleCheckboxChange(category, option.value)}
                              className="hidden"
                            />
                            <div
                              className={`flex items-center gap-4 px-5 py-4 rounded-2xl border-2 transition-all duration-300 group-hover:scale-105 ${isChecked
                                ? 'border-orange-300 bg-gradient-to-r from-orange-50 to-red-50 shadow-lg shadow-orange-100'
                                : "border-gray-200 bg-white hover:bg-gray-50 hover:border-gray-300 hover:shadow-md"
                                }`}
                            >
                              <div
                                className={`w-7 h-7 flex items-center justify-center rounded-full border-2 transition-all duration-300 ${isChecked
                                  ? 'bg-gradient-to-r from-orange-500 to-red-500 border-orange-500 shadow-md'
                                  : "border-gray-300 group-hover:border-gray-400"
                                  }`}
                              >
                                <Check className={`w-4 h-4 text-white transition-all duration-300 ${isChecked ? 'scale-100 opacity-100' : 'scale-0 opacity-0'
                                  }`} />
                              </div>
                              <span className="text-2xl">{option.icon}</span>
                              <span className="font-bold text-lg flex-1">{option.label}</span>
                            </div>
                          </label>
                        );
                      })
                    )}
                  </div>
                </div>
              </div>
            </div>
          );
        })}

        {/* Submit button */}
        <div className={`text-center transition-all duration-1000 ${isLoaded ? 'translate-y-0 opacity-100' : 'translate-y-10 opacity-0'
          }`} style={{ transitionDelay: '1000ms' }}>
          <button
            onClick={handleSubmit}
            disabled={isSubmitting}
            className={`group relative inline-flex items-center gap-3 px-12 py-5 rounded-full font-bold text-lg shadow-2xl transition-all duration-500 overflow-hidden ${isSubmitting
              ? 'bg-gray-400 cursor-not-allowed'
              : 'bg-gradient-to-r from-orange-500 to-red-500 text-white hover:shadow-orange-200 hover:scale-105 active:scale-95'
              }`}
          >
            <div className="absolute inset-0 bg-gradient-to-r from-red-500 to-pink-500 opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
            <div className="relative z-10 flex items-center gap-3">
              {isSubmitting ? (
                <>
                  <div className="w-6 h-6 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                  <span>Saving Preferences...</span>
                </>
              ) : (
                <>
                  <Star className="w-6 h-6 transition-transform duration-500 group-hover:rotate-12" />
                  <span>Save Preferences</span>
                  <ArrowRight className="w-5 h-5 transition-transform duration-500 group-hover:translate-x-1" />
                </>
              )}
            </div>
          </button>

          {getSelectionCount() > 0 && (
            <p className="mt-4 text-gray-600 animate-fade-in">
              Great choices! You've selected {getSelectionCount()} preferences to personalize your experience.
            </p>
          )}
        </div>
      </div>

      <style jsx>{`
        @keyframes fade-in {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        
        .animate-fade-in {
          animation: fade-in 0.5s ease-out;
        }

        /* Hide scrollbar for Chrome, Safari and Opera */
        ::-webkit-scrollbar {
          display: none;
        }

        /* Hide scrollbar for IE, Edge and Firefox */
        body {
          scrollbar-width: none; /* Firefox */
        }
      `}</style>
    </div>
  );
}

