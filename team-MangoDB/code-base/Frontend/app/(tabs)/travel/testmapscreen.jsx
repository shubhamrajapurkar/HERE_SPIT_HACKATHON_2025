import React, { useState, useEffect, useRef } from "react";
import {
  View,
  StyleSheet,
  Alert,
  ActivityIndicator,
  Text,
  Dimensions,
  StatusBar,
  Platform,
  Modal,
  TouchableOpacity,
  ScrollView,
  Animated,
  Image
} from "react-native";
import MapView, {
  Marker,
  Polyline,
  PROVIDER_DEFAULT
} from "react-native-maps";
import { useLocalSearchParams } from "expo-router";
import polyline from '@mapbox/polyline';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';

const { width, height } = Dimensions.get('window');

const ACCENT_COLOR = '#047857'; // Changed accent color
const PRIMARY_TEXT_COLOR = '#333333';
const SECONDARY_TEXT_COLOR = '#666666';
const WHITE = '#FFFFFF';
const LIGHT_GRAY = '#f8f9fa';
const BORDER_COLOR = '#f0f0f0';
const SHADOW_COLOR = '#000000';

const CustomMap = () => {
  const { resp_data } = useLocalSearchParams();
  const main_data = JSON.parse(resp_data);
  console.log("main_data", main_data);

  const [routeCoordinates, setRouteCoordinates] = useState([]);
  const [markers, setMarkers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [mapRegion, setMapRegion] = useState(null);
  const [routeInfo, setRouteInfo] = useState({ duration: 0, distance: 0 });
  const [modalVisible, setModalVisible] = useState(false);
  const [selectedSuggestionIndex, setSelectedSuggestionIndex] = useState(0);
  const [extraStopSuggestions, setExtraStopSuggestions] = useState([]);
  const [mainLocations, setMainLocations] = useState([]);
  const slideAnim = useRef(new Animated.Value(height)).current;

  useEffect(() => {
    processRouteData();
  }, []);

  const openModal = () => {
    setModalVisible(true);
    Animated.timing(slideAnim, {
      toValue: 0,
      duration: 300,
      useNativeDriver: true,
    }).start();
  };

  const closeModal = () => {
    Animated.timing(slideAnim, {
      toValue: height,
      duration: 300,
      useNativeDriver: true,
    }).start(() => setModalVisible(false));
  };

  const processRouteData = async () => {
    try {
      const locations = extractLocations(main_data);
      console.log("Extracted locations:", locations);

      // Store main locations
      const mainLocs = locations.filter(loc => loc.type === 'main');
      setMainLocations(mainLocs);

      // Store extra stop suggestions
      const extraStops = main_data.route.route_locations.find(item => item.type === "extra");
      if (extraStops && extraStops.suggestions) {
        setExtraStopSuggestions(extraStops.suggestions);
      }

      if (locations.length >= 2) {
        setMarkers(locations);
        setMapRegion(calculateMapRegion(locations));
        await fetchRoute(locations);
      } else {
        Alert.alert("Error", "Need at least 2 locations for routing");
        setLoading(false);
      }
    } catch (error) {
      console.error("Error processing route data:", error);
      Alert.alert("Error", "Failed to process route data");
      setLoading(false);
    }
  };

  const extractLocations = (data) => {
    const locations = [];

    if (data.route && data.route.route_locations) {
      data.route.route_locations.forEach((item, index) => {
        if (item.type === "main" && item.location) {
          locations.push({
            latitude: item.location.latitude,
            longitude: item.location.longitude,
            name: item.location.name,
            type: 'main',
            index: index
          });
        } else if (item.type === "extra" && item.suggestions && item.suggestions.length > 0) {
          // Take the first suggestion (zeroth element)
          const suggestion = item.suggestions[0];
          locations.push({
            latitude: suggestion.latitude,
            longitude: suggestion.longitude,
            name: suggestion.name,
            type: 'extra',
            index: index
          });
        }
      });
    }

    return locations;
  };

  const calculateMapRegion = (locations) => {
    if (locations.length === 0) return null;

    let minLat = locations[0].latitude;
    let maxLat = locations[0].latitude;
    let minLng = locations[0].longitude;
    let maxLng = locations[0].longitude;

    locations.forEach(location => {
      minLat = Math.min(minLat, location.latitude);
      maxLat = Math.max(maxLat, location.latitude);
      minLng = Math.min(minLng, location.longitude);
      maxLng = Math.max(maxLng, location.longitude);
    });

    const latDelta = Math.max((maxLat - minLat) * 1.8, 0.02); // Add more padding
    const lngDelta = Math.max((maxLng - minLng) * 1.8, 0.02); // Add more padding

    return {
      latitude: (minLat + maxLat) / 2,
      longitude: (minLng + maxLng) / 2,
      latitudeDelta: latDelta,
      longitudeDelta: lngDelta,
    };
  };

  const fetchRoute = async (locations, extraStopIndex = 0) => {
    try {
      if (locations.length < 2) {
        setLoading(false);
        return;
      }

      // Clone the locations array to avoid mutating the original
      const routeLocations = [...locations];

      // If there's an extra stop, update it with the selected suggestion
      const extraStop = main_data.route.route_locations.find(item => item.type === "extra");
      if (extraStop && extraStop.suggestions && extraStop.suggestions.length > 0) {
        const extraIndex = routeLocations.findIndex(loc => loc.type === "extra");
        if (extraIndex !== -1) {
          routeLocations[extraIndex] = {
            ...routeLocations[extraIndex],
            latitude: extraStop.suggestions[extraStopIndex].latitude,
            longitude: extraStop.suggestions[extraStopIndex].longitude,
            name: extraStop.suggestions[extraStopIndex].name
          };
        }
      }

      const origin = `${routeLocations[0].latitude},${routeLocations[0].longitude}`;
      const destination = `${routeLocations[routeLocations.length - 1].latitude},${routeLocations[routeLocations.length - 1].longitude}`;

      // Format waypoints for HERE API
      const waypoints = routeLocations.slice(1, -1).map(loc => `${loc.latitude},${loc.longitude}`);

      const apiKey = process.env.EXPO_PUBLIC_here_api;

      // Build HERE Routing API URL
      let apiUrl = `https://router.hereapi.com/v8/routes?transportMode=car&origin=${origin}&destination=${destination}&return=polyline,summary&apikey=${apiKey}`;
      waypoints.forEach((wp, idx) => {
        apiUrl += `&via${idx + 1}=${wp}`;
      });

      console.log('HERE API URL:', apiUrl);

      const response = await fetch(apiUrl, {
        method: "GET",
      });

      const data = await response.json();

      if (data.routes && data.routes.length > 0) {
        const route = data.routes[0];
        // HERE polyline is encoded in polyline format
        const polylineStr = route.sections[0].polyline;
        const coordinates = polyline.decode(polylineStr).map(([lat, lng]) => ({
          latitude: lat,
          longitude: lng,
        }));
        setRouteCoordinates(coordinates);

        // Extract route summary
        const summary = route.sections[0].summary;
        setRouteInfo({
          duration: Math.round(summary.duration / 60), // seconds to minutes
          distance: Math.round(summary.length / 1000 * 100) / 100 // meters to km
        });

        // Update markers with the new locations
        setMarkers(routeLocations);

        console.log("Route coordinates loaded:", coordinates.length, "points");
      } else {
        throw new Error('No routes found in response');
      }
    } catch (error) {
      console.error("Error fetching route:", error);
      Alert.alert("Error", "Failed to fetch route");
    } finally {
      setLoading(false);
    }
  };

  const handleSuggestionSelect = (index) => {
    setSelectedSuggestionIndex(index);
    fetchRoute(markers, index);
  };

  const getMarkerColor = (type, index) => {
    if (index === 0) return '#4CAF50'; // Material Green for start
    if (index === markers.length - 1) return '#F44336'; // Material Red for end
    if (type === 'extra') return '#FF9800'; // Material Orange for extra stops
    return ACCENT_COLOR; // Accent color for main waypoints
  };

  const getMarkerTitle = (marker, index) => {
    if (index === 0) return `🏁 Start: ${marker.name}`;
    if (index === markers.length - 1) return `🏆 End: ${marker.name}`;
    if (marker.type === 'extra') return `☕ Stop: ${marker.name}`;
    return `📍 Waypoint: ${marker.name}`;
  };

  const formatDuration = (minutes) => {
    if (minutes < 60) return `${minutes}m`;
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    return `${hours}h ${mins}m`;
  };

  const getLocationIcon = (type) => {
    switch (type) {
      case 'point_of_interest':
        return 'location';
      case 'cafe':
        return 'cafe';
      default:
        return 'location';
    }
  };

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <LinearGradient
          colors={['#1a237e', ACCENT_COLOR]} // Changed gradient colors
          style={styles.loadingGradient}
        >
          <ActivityIndicator size="large" color={WHITE} />
          <Text style={styles.loadingText}>🗺️ Loading your route...</Text>
          <Text style={styles.loadingSubtext}>Finding the best path</Text>
        </LinearGradient>
      </View>
    );
  }

  if (!mapRegion) {
    return (
      <View style={styles.errorContainer}>
        <Text style={styles.errorIcon}>⚠️</Text>
        <Text style={styles.errorText}>Unable to load map data</Text>
        <Text style={styles.errorSubtext}>Please check your internet connection</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor={ACCENT_COLOR} />

      <MapView
        provider={PROVIDER_DEFAULT}
        style={styles.map}
        initialRegion={mapRegion}
        showsUserLocation={true}
        showsMyLocationButton={false}
        showsCompass={true}
        showsScale={true}
        toolbarEnabled={false}
        mapType="standard"
        customMapStyle={customMapStyle}
      >
        {/* Render route polyline with gradient effect */}
        {routeCoordinates.length > 0 && (
          <>
            {/* Shadow polyline */}
            <Polyline
              coordinates={routeCoordinates}
              strokeColor="rgba(0, 0, 0, 0.3)"
              strokeWidth={8}
              lineCap="round"
              lineJoin="round"
            />
            {/* Main route polyline */}
            <Polyline
              coordinates={routeCoordinates}
              strokeColor={ACCENT_COLOR} // Changed polyline color
              strokeWidth={6}
              lineCap="round"
              lineJoin="round"
              geodesic={true}
            />
            {/* Highlight polyline */}
            <Polyline
              coordinates={routeCoordinates}
              strokeColor="#81C784" // Adjusted highlight color
              strokeWidth={3}
              lineCap="round"
              lineJoin="round"
              geodesic={true}
            />
          </>
        )}

        {/* Render markers with custom styling */}
        {markers.map((marker, index) => (
          <Marker
            key={`marker-${index}`}
            coordinate={{
              latitude: marker.latitude,
              longitude: marker.longitude,
            }}
            title={getMarkerTitle(marker, index)}
            description={marker.name}
            pinColor={getMarkerColor(marker.type, index)}
            anchor={{ x: 0.5, y: 1 }}
          >
            <View style={[
              styles.customMarker,
              { backgroundColor: getMarkerColor(marker.type, index) }
            ]}>
              <Ionicons
                name={getLocationIcon(marker.type)}
                size={16}
                color={WHITE}
              />
            </View>
          </Marker>
        ))}
      </MapView>

      {/* Route details button */}
      <TouchableOpacity style={styles.detailsButton} onPress={openModal}>
        <Ionicons name="information-circle" size={24} color={ACCENT_COLOR} />
        <Text style={styles.detailsButtonText}>Route Details</Text>
      </TouchableOpacity>

      <View className="h-10 w-[8rem] bg-white absolute z-10 bottom-1 left-1">
        <View style={{ flexDirection: 'row', alignItems: 'center', gap: 4 }}>
          <View style={{ width: 10, height: 10, backgroundColor: '#4CAF50', borderRadius: 5 }} />
          <Text style={{ fontSize: 12, color: '#333' }}>Start</Text>
          <View style={{ width: 10, height: 10, backgroundColor: '#F44336', borderRadius: 5, marginLeft: 8 }} />
          <Text style={{ fontSize: 12, color: '#333' }}>End</Text>
        </View>
      </View>

      {/* Bottom sheet modal */}
      <Modal
        visible={modalVisible}
        transparent={true}
        animationType="none"
        onRequestClose={closeModal}
      >
        <View style={styles.modalOverlay}>
          <Animated.View
            style={[
              styles.modalContainer,
              { transform: [{ translateY: slideAnim }] }
            ]}
          >
            {/* Handle bar */}
            <View style={styles.handleBarContainer}>
              <View style={styles.handleBar} />
            </View>

            {/* Route summary */}
            <View style={styles.routeSummary}>
              <View style={styles.routeSummaryItem}>
                <Ionicons name="time" size={20} color={ACCENT_COLOR} />
                <Text style={styles.routeSummaryText}>{formatDuration(routeInfo.duration)}</Text>
              </View>
              <View style={styles.routeSummaryItem}>
                <Ionicons name="speedometer" size={20} color={ACCENT_COLOR} />
                <Text style={styles.routeSummaryText}>{routeInfo.distance} km</Text>
              </View>
            </View>

            {/* Main locations */}
            <ScrollView style={styles.locationsContainer}>
              {mainLocations.map((location, index) => (
                <View key={`location-${index}`} style={styles.locationCard}>
                  <View style={styles.locationIcon}>
                    <Ionicons
                      name={index === 0 ? "flag" : index === mainLocations.length - 1 ? "flag" : "location"}
                      size={20}
                      color={index === 0 ? "#4CAF50" : index === mainLocations.length - 1 ? "#F44336" : ACCENT_COLOR}
                    />
                  </View>
                  <View style={styles.locationDetails}>
                    <Text style={styles.locationName}>{location.name.toUpperCase()}</Text>
                    <Text style={styles.locationType}>
                      {index === 0 ? "START POINT" :
                        index === mainLocations.length - 1 ? "DESTINATION" :
                          "WAYPOINT"}
                    </Text>
                  </View>
                </View>
              ))}
            </ScrollView>

            {/* Suggestions carousel */}
            {extraStopSuggestions.length > 0 && (
              <View style={styles.suggestionsContainer}>
                <Text style={styles.sectionTitle}>Stop Options:</Text>
                <ScrollView
                  horizontal
                  showsHorizontalScrollIndicator={false}
                  contentContainerStyle={styles.suggestionsScroll}
                >
                  {extraStopSuggestions.map((suggestion, index) => (
                    <TouchableOpacity
                      key={`suggestion-${index}`}
                      style={[
                        styles.suggestionCard,
                        selectedSuggestionIndex === index && styles.selectedSuggestionCard
                      ]}
                      onPress={() => handleSuggestionSelect(index)}
                    >
                      <View style={styles.suggestionIcon}>
                        <Ionicons name="cafe" size={20} color="#FF9800" />
                      </View>
                      <View style={styles.suggestionTextContainer}>
                        <Text style={styles.suggestionName} numberOfLines={2}>
                          {suggestion.name.split(',')[0]}
                        </Text>
                        <Text style={styles.suggestionDistance}>
                          {index === 0 ? 'DEFAULT' : `${Math.floor(Math.random() * 5) + 1} MIN DETOUR`}
                        </Text>
                      </View>
                    </TouchableOpacity>
                  ))}
                </ScrollView>
              </View>
            )}

            {/* Close button */}
            <TouchableOpacity style={styles.closeButton} onPress={closeModal}>
              <Text style={styles.closeButtonText}>Close</Text>
            </TouchableOpacity>
          </Animated.View>
        </View>
      </Modal>
    </View>
  );
};

// Custom map styling
const customMapStyle = [
  {
    featureType: "water",
    elementType: "geometry",
    stylers: [{ color: "#e9e9e9" }, { lightness: 17 }]
  },
  {
    featureType: "landscape",
    elementType: "geometry",
    stylers: [{ color: "#f5f5f5" }, { lightness: 20 }]
  },
  {
    featureType: "road.highway",
    elementType: "geometry.fill",
    stylers: [{ color: "#ffffff" }, { lightness: 17 }]
  },
  {
    featureType: "road.highway",
    elementType: "geometry.stroke",
    stylers: [{ color: "#ffffff" }, { lightness: 29 }, { weight: 0.2 }]
  },
  {
    featureType: "road.arterial",
    elementType: "geometry",
    stylers: [{ color: "#ffffff" }, { lightness: 18 }]
  },
  {
    featureType: "road.local",
    elementType: "geometry",
    stylers: [{ color: "#ffffff" }, { lightness: 16 }]
  },
  {
    featureType: "poi",
    elementType: "geometry",
    stylers: [{ color: "#f5f5f5" }, { lightness: 21 }]
  }
];

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: LIGHT_GRAY,
  },
  map: {
    flex: 1,
  },
  loadingContainer: {
    flex: 1,
  },
  loadingGradient: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 40,
  },
  loadingText: {
    marginTop: 20,
    fontSize: 20,
    fontWeight: '600',
    color: WHITE,
    textAlign: 'center',
  },
  loadingSubtext: {
    marginTop: 8,
    fontSize: 16,
    color: 'rgba(255, 255, 255, 0.8)',
    textAlign: 'center',
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: LIGHT_GRAY,
    paddingHorizontal: 40,
  },
  errorIcon: {
    fontSize: 48,
    marginBottom: 16,
  },
  errorText: {
    fontSize: 18,
    fontWeight: '600',
    color: '#d32f2f',
    textAlign: 'center',
    marginBottom: 8,
  },
  errorSubtext: {
    fontSize: 14,
    color: SECONDARY_TEXT_COLOR,
    textAlign: 'center',
  },
  detailsButton: {
    position: 'absolute',
    top: Platform.OS === 'ios' ? 60 : 40,
    right: 20,
    backgroundColor: WHITE,
    paddingVertical: 10,
    paddingHorizontal: 15,
    borderRadius: 20,
    flexDirection: 'row',
    alignItems: 'center',
    elevation: 5,
    shadowColor: SHADOW_COLOR,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 4,
  },
  detailsButtonText: {
    marginLeft: 8,
    color: ACCENT_COLOR,
    fontWeight: '600',
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'flex-end',
  },
  modalContainer: {
    backgroundColor: WHITE,
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    paddingHorizontal: 20,
    paddingBottom: 30,
    maxHeight: height * 0.8,
  },
  handleBarContainer: {
    alignItems: 'center',
    paddingVertical: 10,
  },
  handleBar: {
    width: 40,
    height: 5,
    backgroundColor: '#ccc',
    borderRadius: 3,
  },
  routeSummary: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    paddingVertical: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  routeSummaryItem: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  routeSummaryText: {
    marginLeft: 8,
    fontSize: 16,
    fontWeight: '500',
    color: PRIMARY_TEXT_COLOR,
  },
  locationsContainer: {
    marginTop: 15,
    maxHeight: height * 0.4,
  },
  locationCard: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: BORDER_COLOR,
  },
  locationIcon: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: '#f5f5f5',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 15,
  },
  locationDetails: {
    flex: 1,
  },
  locationName: {
    fontSize: 16,
    fontWeight: '500',
    marginBottom: 4,
  },
  locationType: {
    fontSize: 14,
    color: SECONDARY_TEXT_COLOR,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 10,
    color: PRIMARY_TEXT_COLOR,
  },
  suggestionsContainer: {
    marginTop: 15,
  },
  suggestionsScroll: {
    paddingBottom: 10,
  },
  suggestionCard: {
    width: 180,
    height: 80,
    backgroundColor: '#f5f5f5',
    borderRadius: 10,
    padding: 10,
    marginRight: 10,
    flexDirection: 'row',
    alignItems: 'center',
  },
  selectedSuggestionCard: {
    backgroundColor: '#e8f5e9', // Adjusted selected suggestion background
    borderColor: ACCENT_COLOR,
    borderWidth: 1,
  },
  suggestionIcon: {
    marginRight: 10,
  },
  suggestionTextContainer: {
    flex: 1,
  },
  suggestionName: {
    fontSize: 14,
    fontWeight: '500',
    marginBottom: 5,
  },
  suggestionDistance: {
    fontSize: 12,
    color: SECONDARY_TEXT_COLOR,
  },
  closeButton: {
    backgroundColor: ACCENT_COLOR,
    paddingVertical: 12,
    borderRadius: 8,
    marginTop: 15,
    alignItems: 'center',
  },
  closeButtonText: {
    color: WHITE,
    fontWeight: '600',
    fontSize: 16,
  },
  customMarker: {
    width: 36,
    height: 36,
    borderRadius: 18,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 3,
    borderColor: WHITE,
    elevation: 4,
    shadowColor: SHADOW_COLOR,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 4,
  },
});

const decodePolyline = (encoded) => {
  const points = [];
  let index = 0;
  const len = encoded.length;
  let lat = 0;
  let lng = 0;

  while (index < len) {
    let b;
    let shift = 0;
    let result = 0;

    do {
      b = encoded.charCodeAt(index++) - 63;
      result |= (b & 0x1f) << shift;
      shift += 5;
    } while (b >= 0x20);

    const dlat = ((result & 1) ? ~(result >> 1) : (result >> 1));
    lat += dlat;

    shift = 0;
    result = 0;

    do {
      b = encoded.charCodeAt(index++) - 63;
      result |= (b & 0x1f) << shift;
      shift += 5;
    } while (b >= 0x20);

    const dlng = ((result & 1) ? ~(result >> 1) : (result >> 1));
    lng += dlng;

    points.push({
      latitude: lat / 1e5,
      longitude: lng / 1e5
    });
  }

  return points;
};

export default CustomMap;