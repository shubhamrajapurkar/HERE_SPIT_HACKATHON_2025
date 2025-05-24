export const calculateWeightedScore = (restaurant) => {
  const weights = {
    rating: 0.4,
    quality: 0.3,
    hygiene: 0.3
  };

  const hygieneScores = {
    'very good': 10,
    'good': 7.5,
    'acceptable': 5,
    'needs improvement': 2.5
  };

  const score = (restaurant.rating * weights.rating) +
                (restaurant.quality * weights.quality) +
                (hygieneScores[restaurant.hygiene] * weights.hygiene);
  
  return score / 10; // Normalize to 0-1
};

export const getDietaryColor = (restaurant, score) => {
  const { dietary_options } = restaurant;
  let baseColor;
  
  // Base color by dietary type
  if (dietary_options.vegan) {
    baseColor = [46, 204, 113]; // Green
  } else if (dietary_options.vegetarian) {
    baseColor = [241, 196, 15]; // Yellow
  } else if (dietary_options.jain) {
    baseColor = [155, 89, 182]; // Purple
  } else {
    baseColor = [231, 76, 60]; // Red (non-veg)
  }

  // Adjust intensity based on weighted score
  const intensity = 0.3 + (score * 0.7); // Min 30% intensity
  
  return baseColor.map(channel => Math.round(channel * intensity));
};