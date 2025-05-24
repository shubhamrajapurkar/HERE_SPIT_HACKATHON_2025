import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
  TextInput,
} from 'react-native';
import * as Location from 'expo-location';
import HereApiService from '../services/HereApiService';

const RouteDetailsScreen = ({ navigation }) => {
  const [sourceLocation, setSourceLocation] = useState('');
  const [destinationLocation, setDestinationLocation] = useState('');
  const [routes, setRoutes] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedTransportMode, setSelectedTransportMode] = useState('car');
  const [currentLocation, setCurrentLocation] = useState(null);

  const transportModes = [
    { key: 'car', label: 'Car', icon: '🚗' },
    { key: 'truck', label: 'Truck', icon: '🚛' },
    { key: 'pedestrian', label: 'Walk', icon: '🚶' },
    { key: 'bicycle', label: 'Bike', icon: '🚴' },
  ];

  useEffect(() => {
    getCurrentLocation();
  }, []);

  const getCurrentLocation = async () => {
    try {
      const { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') {
        console.log('Permission to access location was denied');
        return;
      }

      const location = await Location.getCurrentPositionAsync({
        accuracy: Location.Accuracy.High,
      });
      
      const { latitude, longitude } = location.coords;
      setCurrentLocation({ lat: latitude, lng: longitude });
      setSourceLocation(`${latitude.toFixed(4)}, ${longitude.toFixed(4)}`);
    } catch (error) {
      console.error('Error getting location:', error);
    }
  };

  const calculateRoutes = async () => {
    if (!sourceLocation.trim() || !destinationLocation.trim()) {
      Alert.alert('Error', 'Please enter both source and destination');
      return;
    }

    // For demo purposes, using current location and a nearby destination
    // In a real app, you would geocode the addresses first
    if (!currentLocation) {
      Alert.alert('Error', 'Current location not available');
      return;
    }

    // Parse destination coordinates (simplified - in real app use geocoding)
    const destination = { lat: currentLocation.lat + 0.01, lng: currentLocation.lng + 0.01 };

    setLoading(true);
    try {
      // Get main route
      const mainRouteResponse = await HereApiService.calculateRoute(
        currentLocation,
        destination,
        selectedTransportMode
      );

      // Get alternative routes
      const altRoutesResponse = await HereApiService.getAlternativeRoutes(
        currentLocation,
        destination,
        selectedTransportMode,
        3
      );

      if (mainRouteResponse.success && altRoutesResponse.success) {
        const allRoutes = altRoutesResponse.data.routes.map((route, index) => ({
          id: index,
          ...route,
          isMain: index === 0,
        }));
        setRoutes(allRoutes);
      }
    } catch (error) {
      Alert.alert('Error', error.message);
    } finally {
      setLoading(false);
    }
  };

  const formatDuration = (seconds) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    if (hours > 0) {
      return `${hours}h ${minutes}m`;
    }
    return `${minutes}m`;
  };

  const formatDistance = (meters) => {
    if (meters >= 1000) {
      return `${(meters / 1000).toFixed(1)} km`;
    }
    return `${meters} m`;
  };

  const RouteCard = ({ route, index }) => (
    <View style={[styles.routeCard, route.isMain && styles.mainRoute]}>
      <View style={styles.routeHeader}>
        <Text style={styles.routeTitle}>
          {route.isMain ? 'Main Route' : `Alternative ${index}`}
        </Text>
        {route.isMain && <Text style={styles.mainBadge}>FASTEST</Text>}
      </View>
      
      <View style={styles.routeStats}>
        <View style={styles.statItem}>
          <Text style={styles.statLabel}>Duration</Text>
          <Text style={styles.statValue}>
            {formatDuration(route.sections[0].summary.duration)}
          </Text>
        </View>
        
        <View style={styles.statItem}>
          <Text style={styles.statLabel}>Distance</Text>
          <Text style={styles.statValue}>
            {formatDistance(route.sections[0].summary.length)}
          </Text>
        </View>
        
        <View style={styles.statItem}>
          <Text style={styles.statLabel}>Traffic</Text>
          <Text style={styles.statValue}>
            {route.sections[0].summary.trafficTime 
              ? formatDuration(route.sections[0].summary.trafficTime)
              : 'No data'
            }
          </Text>
        </View>
      </View>

      {route.sections[0].actions && (
        <ScrollView style={styles.instructionsContainer} nestedScrollEnabled>
          <Text style={styles.instructionsTitle}>Turn-by-turn directions:</Text>
          {route.sections[0].actions.map((action, actionIndex) => (
            <View key={actionIndex} style={styles.instruction}>
              <Text style={styles.instructionText}>
                {actionIndex + 1}. {action.instruction}
              </Text>
              {action.length && (
                <Text style={styles.instructionDistance}>
                  {formatDistance(action.length)}
                </Text>
              )}
            </View>
          ))}
        </ScrollView>
      )}

      <TouchableOpacity
        style={styles.selectRouteButton}
        onPress={() => {
          // Navigate to map with this route selected
          navigation.navigate('Map', { selectedRoute: route });
        }}
      >
        <Text style={styles.selectRouteText}>View on Map</Text>
      </TouchableOpacity>
    </View>
  );

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Route Planning</Text>
      </View>

      <View style={styles.inputContainer}>
        <Text style={styles.inputLabel}>Source Location</Text>
        <TextInput
          style={styles.input}
          placeholder="Enter source location or use current location"
          value={sourceLocation}
          onChangeText={setSourceLocation}
        />
        
        <TouchableOpacity
          style={styles.useCurrentLocationButton}
          onPress={getCurrentLocation}
        >
          <Text style={styles.useCurrentLocationText}>📍 Use Current Location</Text>
        </TouchableOpacity>
        
        <Text style={styles.inputLabel}>Destination Location</Text>
        <TextInput
          style={styles.input}
          placeholder="Enter destination location"
          value={destinationLocation}
          onChangeText={setDestinationLocation}
        />

        <View style={styles.transportModeContainer}>
          <Text style={styles.transportModeLabel}>Transport Mode:</Text>
          <ScrollView horizontal showsHorizontalScrollIndicator={false}>
            {transportModes.map((mode) => (
              <TouchableOpacity
                key={mode.key}
                style={[
                  styles.modeButton,
                  selectedTransportMode === mode.key && styles.selectedModeButton
                ]}
                onPress={() => setSelectedTransportMode(mode.key)}
              >
                <Text style={styles.modeIcon}>{mode.icon}</Text>
                <Text style={styles.modeLabel}>{mode.label}</Text>
              </TouchableOpacity>
            ))}
          </ScrollView>
        </View>

        <TouchableOpacity
          style={styles.calculateButton}
          onPress={calculateRoutes}
          disabled={loading}
        >
          {loading ? (
            <ActivityIndicator color="white" />
          ) : (
            <Text style={styles.calculateButtonText}>Calculate Routes</Text>
          )}
        </TouchableOpacity>
      </View>

      {routes.length > 0 && (
        <View style={styles.routesContainer}>
          <Text style={styles.routesTitle}>Available Routes</Text>
          {routes.map((route, index) => (
            <RouteCard key={route.id} route={route} index={index} />
          ))}
        </View>
      )}
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    backgroundColor: '#007AFF',
    padding: 20,
    paddingTop: 60,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: 'white',
    textAlign: 'center',
  },
  inputContainer: {
    padding: 20,
    backgroundColor: 'white',
    margin: 10,
    borderRadius: 10,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  inputLabel: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 5,
    color: '#333',
  },
  input: {
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 8,
    padding: 12,
    marginBottom: 10,
    fontSize: 16,
  },
  useCurrentLocationButton: {
    backgroundColor: '#4CAF50',
    padding: 12,
    borderRadius: 8,
    marginBottom: 15,
    alignItems: 'center',
  },
  useCurrentLocationText: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: 14,
  },
  transportModeContainer: {
    marginBottom: 20,
  },
  transportModeLabel: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  modeButton: {
    alignItems: 'center',
    padding: 10,
    marginRight: 10,
    borderRadius: 8,
    backgroundColor: '#f0f0f0',
    minWidth: 60,
  },
  selectedModeButton: {
    backgroundColor: '#007AFF',
  },
  modeIcon: {
    fontSize: 20,
    marginBottom: 5,
  },
  modeLabel: {
    fontSize: 12,
    fontWeight: 'bold',
  },
  calculateButton: {
    backgroundColor: '#007AFF',
    padding: 15,
    borderRadius: 8,
    alignItems: 'center',
  },
  calculateButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  routesContainer: {
    padding: 10,
  },
  routesTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 15,
    textAlign: 'center',
  },
  routeCard: {
    backgroundColor: 'white',
    borderRadius: 10,
    padding: 15,
    marginBottom: 15,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  mainRoute: {
    borderWidth: 2,
    borderColor: '#007AFF',
  },
  routeHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  routeTitle: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  mainBadge: {
    backgroundColor: '#007AFF',
    color: 'white',
    padding: 5,
    borderRadius: 5,
    fontSize: 12,
    fontWeight: 'bold',
  },
  routeStats: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginBottom: 15,
  },
  statItem: {
    alignItems: 'center',
  },
  statLabel: {
    fontSize: 12,
    color: '#666',
    marginBottom: 5,
  },
  statValue: {
    fontSize: 16,
    fontWeight: 'bold',
  },
  instructionsContainer: {
    maxHeight: 150,
    marginBottom: 15,
  },
  instructionsTitle: {
    fontSize: 14,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  instruction: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 5,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  instructionText: {
    flex: 1,
    fontSize: 12,
  },
  instructionDistance: {
    fontSize: 12,
    color: '#666',
    marginLeft: 10,
  },
  selectRouteButton: {
    backgroundColor: '#007AFF',
    padding: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  selectRouteText: {
    color: 'white',
    fontWeight: 'bold',
  },
});

export default RouteDetailsScreen;