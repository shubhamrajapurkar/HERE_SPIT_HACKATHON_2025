import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  StyleSheet,
  Alert,
  ActivityIndicator,
  Text,
  TouchableOpacity,
  Modal,
  ScrollView,
  Dimensions,
  TextInput,
  KeyboardAvoidingView,
  Platform,
} from 'react-native';
import MapView, { Marker, Polyline, PROVIDER_GOOGLE } from 'react-native-maps';
import * as Location from 'expo-location';
import HereApiService from '../services/HereApiService';

const { width, height } = Dimensions.get('window');

const MapScreen = () => {
  const [currentLocation, setCurrentLocation] = useState(null);
  const [destination, setDestination] = useState(null);
  const [route, setRoute] = useState(null);
  const [alternativeRoutes, setAlternativeRoutes] = useState([]);
  const [trafficIncidents, setTrafficIncidents] = useState([]);
  const [loading, setLoading] = useState(false);
  const [mapRegion, setMapRegion] = useState({
    latitude: 37.78825,
    longitude: -122.4324,
    latitudeDelta: 0.0922,
    longitudeDelta: 0.0421,
  });
  const [showRouteOptions, setShowRouteOptions] = useState(false);
  const [showDestinationInput, setShowDestinationInput] = useState(false);
  const [selectedTransportMode, setSelectedTransportMode] = useState('car');
  const [destinationAddress, setDestinationAddress] = useState('');
  const [sourceAddress, setSourceAddress] = useState('');
  
  const mapRef = useRef(null);

  useEffect(() => {
    requestLocationPermission();
  }, []);

  const requestLocationPermission = async () => {
    try {
      const { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permission denied', 'Location permission is required to use this feature.');
        return;
      }
      getCurrentLocation();
    } catch (error) {
      console.error('Permission error:', error);
      Alert.alert('Error', 'Failed to request location permission');
    }
  };

  const getCurrentLocation = async () => {
    try {
      const location = await Location.getCurrentPositionAsync({
        accuracy: Location.Accuracy.High,
      });
      
      const { latitude, longitude } = location.coords;
      const locationObj = { lat: latitude, lng: longitude };
      setCurrentLocation(locationObj);
      setMapRegion({
        latitude,
        longitude,
        latitudeDelta: 0.01,
        longitudeDelta: 0.01,
      });
      
      // Set source address
      setSourceAddress(`${latitude.toFixed(4)}, ${longitude.toFixed(4)}`);
      
      // Load traffic incidents for current location
      loadTrafficIncidents(latitude, longitude);
    } catch (error) {
      console.error('Location error:', error);
      Alert.alert('Error', 'Unable to get current location');
    }
  };

  const loadTrafficIncidents = async (latitude, longitude) => {
    try {
      const response = await HereApiService.getTrafficIncidents(latitude, longitude);
      if (response.success && response.data.TRAFFICITEMS) {
        setTrafficIncidents(response.data.TRAFFICITEMS.TRAFFICITEM || []);
      }
    } catch (error) {
      console.log('Traffic incidents error:', error.message);
    }
  };

  const onMapPress = (event) => {
    const { latitude, longitude } = event.nativeEvent.coordinate;
    setDestination({ lat: latitude, lng: longitude });
    setDestinationAddress(`${latitude.toFixed(4)}, ${longitude.toFixed(4)}`);
  };

  const setDestinationFromInput = () => {
    if (!destinationAddress.trim()) {
      Alert.alert('Error', 'Please enter a destination address');
      return;
    }

    // For demo purposes, convert address to coordinates
    // In a real app, you would use geocoding service
    if (destinationAddress.includes(',')) {
      const coords = destinationAddress.split(',');
      if (coords.length === 2) {
        const lat = parseFloat(coords[0].trim());
        const lng = parseFloat(coords[1].trim());
        if (!isNaN(lat) && !isNaN(lng)) {
          setDestination({ lat, lng });
          setMapRegion({
            latitude: lat,
            longitude: lng,
            latitudeDelta: 0.01,
            longitudeDelta: 0.01,
          });
          setShowDestinationInput(false);
          return;
        }
      }
    }

    // If not coordinates, use current location + offset for demo
    if (currentLocation) {
      const offsetLat = currentLocation.lat + (Math.random() - 0.5) * 0.02;
      const offsetLng = currentLocation.lng + (Math.random() - 0.5) * 0.02;
      setDestination({ lat: offsetLat, lng: offsetLng });
      setDestinationAddress(`${offsetLat.toFixed(4)}, ${offsetLng.toFixed(4)}`);
      setShowDestinationInput(false);
    } else {
      Alert.alert('Error', 'Current location not available. Please enter coordinates in format: lat, lng');
    }
  };

  const calculateRoute = async () => {
    if (!currentLocation || !destination) {
      Alert.alert('Error', 'Please select both source and destination');
      return;
    }

    setLoading(true);
    try {
      const response = await HereApiService.calculateRoute(
        currentLocation,
        destination,
        selectedTransportMode
      );
      
      if (response.success && response.data.routes && response.data.routes.length > 0) {
        const routeData = response.data.routes[0];
        const decodedPolyline = decodePolyline(routeData.sections[0].polyline);
        setRoute({
          coordinates: decodedPolyline,
          summary: routeData.sections[0].summary
        });
        
        // Fit map to show the route
        if (mapRef.current) {
          mapRef.current.fitToCoordinates(decodedPolyline, {
            edgePadding: { top: 50, right: 50, bottom: 50, left: 50 },
            animated: true,
          });
        }
      }
    } catch (error) {
      Alert.alert('Error', error.message);
    } finally {
      setLoading(false);
    }
  };

  const getAlternativeRoutes = async () => {
    if (!currentLocation || !destination) {
      Alert.alert('Error', 'Please select both source and destination');
      return;
    }

    setLoading(true);
    try {
      console.log(`Fetching alternative routes from ${currentLocation} to ${destination} with transport mode ${selectedTransportMode}`);
      
      const response = await HereApiService.getAlternativeRoutes(
        currentLocation,
        destination,
        selectedTransportMode,
        3
      );
      console.log(destination,currentLocation);
      
      if (response.success && response.data.routes && response.data.routes.length > 0) {
        const routes = response.data.routes.map((routeData, index) => ({
          id: index,
          coordinates: decodePolyline(routeData.sections[0].polyline),
          summary: routeData.sections[0].summary,
          color: index === 0 ? '#007AFF' : '#FF9500'
        }));
        setAlternativeRoutes(routes);
        
        // Fit map to show all routes
        if (mapRef.current && routes.length > 0) {
          const allCoordinates = routes.flatMap(route => route.coordinates);
          mapRef.current.fitToCoordinates(allCoordinates, {
            edgePadding: { top: 50, right: 50, bottom: 50, left: 50 },
            animated: true,
          });
        }
      }
    } catch (error) {
      Alert.alert('Error', error.message);
    } finally {
      setLoading(false);
    }
  };

  const findEmergencyRoute = async (emergencyType) => {
    if (!currentLocation) {
      Alert.alert('Error', 'Current location not available');
      return;
    }

    setLoading(true);
    try {
      const response = await HereApiService.getEmergencyRoute(currentLocation, emergencyType);
      
      if (response.success && response.route.routes && response.route.routes.length > 0) {
        const routeData = response.route.routes[0];
        const decodedPolyline = decodePolyline(routeData.sections[0].polyline);
        
        setRoute({
          coordinates: decodedPolyline,
          summary: routeData.sections[0].summary
        });
        
        const facilityLocation = {
          lat: response.facility.position.lat,
          lng: response.facility.position.lng
        };
        setDestination(facilityLocation);
        setDestinationAddress(`${facilityLocation.lat.toFixed(4)}, ${facilityLocation.lng.toFixed(4)}`);
        
        // Fit map to show the route
        if (mapRef.current) {
          mapRef.current.fitToCoordinates(decodedPolyline, {
            edgePadding: { top: 50, right: 50, bottom: 50, left: 50 },
            animated: true,
          });
        }
        
        Alert.alert(
          'Emergency Route Found',
          `Nearest ${emergencyType}: ${response.facility.title}\nDistance: ${(routeData.sections[0].summary.length / 1000).toFixed(1)} km\nDuration: ${Math.round(routeData.sections[0].summary.duration / 60)} minutes`
        );
      }
    } catch (error) {
      Alert.alert('Error', error.message);
    } finally {
      setLoading(false);
    }
  };

  // Polyline decoder for HERE polylines
  const decodePolyline = (encoded) => {
     
    const points = [];
    let index = 0;
    const len = encoded.length;
    let lat = 0;
    let lng = 0;

    while (index < len) {
      let b, shift = 0, result = 0;
      do {
        b = encoded.charCodeAt(index++) - 63;
        result |= (b & 0x1f) << shift;
        shift += 5;
      } while (b >= 0x20);
      const dlat = ((result & 1) !== 0 ? ~(result >> 1) : (result >> 1));
      lat += dlat;

      shift = 0;
      result = 0;
      do {
        b = encoded.charCodeAt(index++) - 63;
        result |= (b & 0x1f) << shift;
        shift += 5;
      } while (b >= 0x20);
      const dlng = ((result & 1) !== 0 ? ~(result >> 1) : (result >> 1));
      lng += dlng;

      points.push({
        latitude: lat / 1e5,
        longitude: lng / 1e5,
      });
    }

    return points;
  };

  const clearRoute = () => {
    setRoute(null);
    setAlternativeRoutes([]);
    setDestination(null);
    setDestinationAddress('');
  };

  const transportModes = [
    { key: 'car', label: 'Car', icon: '🚗' },
    { key: 'truck', label: 'Truck', icon: '🚛' },
    { key: 'pedestrian', label: 'Walk', icon: '🚶' },
    { key: 'bicycle', label: 'Bike', icon: '🚴' },
  ];

  return (
    <View style={styles.container}>
      <MapView
        ref={mapRef}
        provider={PROVIDER_GOOGLE}
        style={styles.map}
        region={mapRegion}
        onPress={onMapPress}
        showsUserLocation={true}
        showsMyLocationButton={true}
        showsTraffic={true}
      >
        {/* Current Location Marker */}
        {currentLocation && (
          <Marker
            coordinate={{
              latitude: currentLocation.lat,
              longitude: currentLocation.lng,
            }}
            title="Your Location"
            pinColor="blue"
          />
        )}

        {/* Destination Marker */}
        {destination && (
          <Marker
            coordinate={{
              latitude: destination.lat,
              longitude: destination.lng,
            }}
            title="Destination"
            pinColor="red"
          />
        )}

        {/* Main Route */}
        {route && (
          <Polyline
            coordinates={route.coordinates}
            strokeColor="#007AFF"
            strokeWidth={4}
          />
        )}

        {/* Alternative Routes */}
        {alternativeRoutes.map((altRoute) => (
          <Polyline
            key={altRoute.id}
            coordinates={altRoute.coordinates}
            strokeColor={altRoute.color}
            strokeWidth={3}
            strokePattern={altRoute.id > 0 ? [10, 5] : undefined}
          />
        ))}

        {/* Traffic Incidents */}
        {trafficIncidents.map((incident, index) => (
          <Marker
            key={index}
            coordinate={{
              latitude: parseFloat(incident.LOCATION.GEOLOC.ORIGIN.LATITUDE),
              longitude: parseFloat(incident.LOCATION.GEOLOC.ORIGIN.LONGITUDE),
            }}
            title="Traffic Incident"
            description={incident.TRAFFICITEMDETAIL.ROADCLOSED ? 'Road Closed' : 'Traffic Alert'}
            pinColor="orange"
          />
        ))}
      </MapView>

      {/* Destination Input Panel */}
      <View style={styles.destinationPanel}>
        <View style={styles.addressContainer}>
          <View style={styles.addressRow}>
            <Text style={styles.addressLabel}>From:</Text>
            <Text style={styles.addressText}>{sourceAddress || 'Getting location...'}</Text>
          </View>
          <View style={styles.addressRow}>
            <Text style={styles.addressLabel}>To:</Text>
            <TouchableOpacity 
              style={styles.destinationInput}
              onPress={() => setShowDestinationInput(true)}
            >
              <Text style={[styles.destinationText, !destinationAddress && styles.placeholderText]}>
                {destinationAddress || 'Tap to set destination'}
              </Text>
            </TouchableOpacity>
          </View>
        </View>
        
        <TouchableOpacity
          style={styles.swapButton}
          onPress={() => {
            // Swap source and destination logic can be added here
            Alert.alert('Info', 'Swap functionality can be added here');
          }}
        >
          <Text style={styles.swapButtonText}>⇅</Text>
        </TouchableOpacity>
      </View>

      {/* Route Information */}
      {route && (
        <View style={styles.routeInfo}>
          <Text style={styles.routeText}>
            Distance: {(route.summary.length / 1000).toFixed(1)} km
          </Text>
          <Text style={styles.routeText}>
            Duration: {Math.round(route.summary.duration / 60)} min
          </Text>
        </View>
      )}

      {/* Control Buttons */}
      <View style={styles.controls}>
        <TouchableOpacity
          style={styles.button}
          onPress={() => setShowRouteOptions(true)}
        >
          <Text style={styles.buttonText}>Route Options</Text>
        </TouchableOpacity>
        
        <TouchableOpacity
          style={styles.button}
          onPress={calculateRoute}
          disabled={loading}
        >
          <Text style={styles.buttonText}>Calculate Route</Text>
        </TouchableOpacity>
        
        <TouchableOpacity
          style={styles.button}
          onPress={getAlternativeRoutes}
          disabled={loading}
        >
          <Text style={styles.buttonText}>Alternatives</Text>
        </TouchableOpacity>
        
        <TouchableOpacity
          style={styles.clearButton}
          onPress={clearRoute}
        >
          <Text style={styles.buttonText}>Clear</Text>
        </TouchableOpacity>
      </View>

      {/* Emergency Buttons */}
      <View style={styles.emergencyControls}>
        <TouchableOpacity
          style={[styles.emergencyButton, { backgroundColor: '#FF3B30' }]}
          onPress={() => findEmergencyRoute('hospital')}
        >
          <Text style={styles.emergencyText}>🏥 Hospital</Text>
        </TouchableOpacity>
        
        <TouchableOpacity
          style={[styles.emergencyButton, { backgroundColor: '#007AFF' }]}
          onPress={() => findEmergencyRoute('police')}
        >
          <Text style={styles.emergencyText}>👮 Police</Text>
        </TouchableOpacity>
        
        <TouchableOpacity
          style={[styles.emergencyButton, { backgroundColor: '#FF9500' }]}
          onPress={() => findEmergencyRoute('fire')}
        >
          <Text style={styles.emergencyText}>🚒 Fire</Text>
        </TouchableOpacity>
      </View>

      {/* Loading Indicator */}
      {loading && (
        <View style={styles.loadingOverlay}>
          <ActivityIndicator size="large" color="#007AFF" />
          <Text style={styles.loadingText}>Calculating route...</Text>
        </View>
      )}

      {/* Destination Input Modal */}
      <Modal
        visible={showDestinationInput}
        transparent={true}
        animationType="slide"
      >
        <KeyboardAvoidingView
          style={styles.modalContainer}
          behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        >
          <View style={styles.destinationModalContent}>
            <Text style={styles.modalTitle}>Set Destination</Text>
            
            <View style={styles.inputContainer}>
              <Text style={styles.inputLabel}>Enter destination address or coordinates:</Text>
              <TextInput
                style={styles.textInput}
                placeholder="e.g., 37.7749, -122.4194 or address"
                value={destinationAddress}
                onChangeText={setDestinationAddress}
                multiline={false}
                autoFocus={true}
              />
              
              <Text style={styles.inputHint}>
                💡 Tip: Enter coordinates as "latitude, longitude" or tap on the map to set destination
              </Text>
            </View>
            
            <View style={styles.modalButtons}>
              <TouchableOpacity
                style={styles.modalCancelButton}
                onPress={() => setShowDestinationInput(false)}
              >
                <Text style={styles.modalCancelButtonText}>Cancel</Text>
              </TouchableOpacity>
              
              <TouchableOpacity
                style={styles.modalSetButton}
                onPress={setDestinationFromInput}
              >
                <Text style={styles.modalSetButtonText}>Set Destination</Text>
              </TouchableOpacity>
            </View>
            
            <TouchableOpacity
              style={styles.useCurrentLocationButton}
              onPress={() => {
                if (currentLocation) {
                  const offsetLat = currentLocation.lat + 0.01;
                  const offsetLng = currentLocation.lng + 0.01;
                  setDestinationAddress(`${offsetLat.toFixed(4)}`,` ${offsetLng.toFixed(4)}`);
                }
              }}
            >
              <Text style={styles.useCurrentLocationText}>📍 Use Nearby Location</Text>
            </TouchableOpacity>
          </View>
        </KeyboardAvoidingView>
      </Modal>

      {/* Route Options Modal */}
      <Modal
        visible={showRouteOptions}
        transparent={true}
        animationType="slide"
      >
        <View style={styles.modalContainer}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>Transport Mode</Text>
            <ScrollView>
              {transportModes.map((mode) => (
                <TouchableOpacity
                  key={mode.key}
                  style={[
                    styles.modeButton,
                    selectedTransportMode === mode.key && styles.selectedMode
                  ]}
                  onPress={() => {
                    setSelectedTransportMode(mode.key);
                    setShowRouteOptions(false);
                  }}
                >
                  <Text style={styles.modeIcon}>{mode.icon}</Text>
                  <Text style={styles.modeLabel}>{mode.label}</Text>
                </TouchableOpacity>
              ))}
            </ScrollView>
            <TouchableOpacity
              style={styles.closeButton}
              onPress={() => setShowRouteOptions(false)}
            >
              <Text style={styles.closeButtonText}>Close</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  map: {
    flex: 1,
  },
  destinationPanel: {
    position: 'absolute',
    top: 50,
    left: 20,
    right: 20,
    backgroundColor: 'white',
    borderRadius: 10,
    padding: 15,
    flexDirection: 'row',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  addressContainer: {
    flex: 1,
  },
  addressRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  addressLabel: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#333',
    width: 40,
  },
  addressText: {
    fontSize: 14,
    color: '#666',
    flex: 1,
  },
  destinationInput: {
    flex: 1,
    backgroundColor: '#f5f5f5',
    padding: 8,
    borderRadius: 5,
    borderWidth: 1,
    borderColor: '#ddd',
  },
  destinationText: {
    fontSize: 14,
    color: '#333',
  },
  placeholderText: {
    color: '#999',
    fontStyle: 'italic',
  },
  swapButton: {
    marginLeft: 10,
    backgroundColor: '#007AFF',
    width: 30,
    height: 30,
    borderRadius: 15,
    justifyContent: 'center',
    alignItems: 'center',
  },
  swapButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  routeInfo: {
    position: 'absolute',
    top: 140,
    left: 20,
    right: 20,
    backgroundColor: 'rgba(255, 255, 255, 0.9)',
    padding: 10,
    borderRadius: 8,
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  routeText: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#333',
  },
  controls: {
    position: 'absolute',
    bottom: 120,
    left: 20,
    right: 20,
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  button: {
    backgroundColor: '#007AFF',
    padding: 12,
    borderRadius: 8,
    flex: 1,
    marginHorizontal: 2,
  },
  clearButton: {
    backgroundColor: '#FF3B30',
    padding: 12,
    borderRadius: 8,
    flex: 1,
    marginHorizontal: 2,
  },
  buttonText: {
    color: 'white',
    textAlign: 'center',
    fontSize: 12,
    fontWeight: 'bold',
  },
  emergencyControls: {
    position: 'absolute',
    bottom: 60,
    left: 20,
    right: 20,
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  emergencyButton: {
    padding: 12,
    borderRadius: 8,
    flex: 1,
    marginHorizontal: 2,
  },
  emergencyText: {
    color: 'white',
    textAlign: 'center',
    fontSize: 12,
    fontWeight: 'bold',
  },
  loadingOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    color: 'white',
    marginTop: 10,
    fontSize: 16,
  },
  modalContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
  },
  destinationModalContent: {
    backgroundColor: 'white',
    padding: 20,
    borderRadius: 10,
    width: width * 0.9,
    maxHeight: height * 0.7,
  },
  modalContent: {
    backgroundColor: 'white',
    padding: 20,
    borderRadius: 10,
    width: width * 0.8,
    maxHeight: height * 0.6,
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 20,
    textAlign: 'center',
  },
  inputContainer: {
    marginBottom: 20,
  },
  inputLabel: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 10,
    color: '#333',
  },
  textInput: {
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 8,
    padding: 12,
    fontSize: 16,
    marginBottom: 10,
  },
  inputHint: {
    fontSize: 12,
    color: '#666',
    fontStyle: 'italic',
  },
  modalButtons: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 15,
  },
  modalCancelButton: {
    backgroundColor: '#ccc',
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 8,
    flex: 0.45,
  },
  modalCancelButtonText: {
    textAlign: 'center',
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
  },
  modalSetButton: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 8,
    flex: 0.45,
  },
  modalSetButtonText: {
    textAlign: 'center',
    fontSize: 16,
    fontWeight: 'bold',
    color: 'white',
  },
  useCurrentLocationButton: {
    backgroundColor: '#4CAF50',
    padding: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  useCurrentLocationText: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: 14,
  },
  modeButton: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 15,
    borderRadius: 8,
    marginBottom: 10,
    backgroundColor: '#f0f0f0',
  },
  selectedMode: {
    backgroundColor: '#007AFF',
  },
  modeIcon: {
    fontSize: 24,
    marginRight: 15,
  },
  modeLabel: {
    fontSize: 16,
    fontWeight: 'bold',
  },
  closeButton: {
    backgroundColor: '#007AFF',
    padding: 15,
    borderRadius: 8,
    marginTop: 20,
  },
  closeButtonText: {
    color: 'white',
    textAlign: 'center',
    fontSize: 16,
    fontWeight: 'bold',
  },
});

export default MapScreen;

