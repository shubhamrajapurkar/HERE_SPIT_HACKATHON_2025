import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
  RefreshControl,
} from 'react-native';
import * as Location from 'expo-location';
import HereApiService from '../services/HereApiService';

const TrafficScreen = () => {
  const [trafficIncidents, setTrafficIncidents] = useState([]);
  const [loading, setLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [currentLocation, setCurrentLocation] = useState(null);
  const [selectedRadius, setSelectedRadius] = useState(5000);

  const radiusOptions = [
    { value: 1000, label: '1 km' },
    { value: 2000, label: '2 km' },
    { value: 5000, label: '5 km' },
    { value: 10000, label: '10 km' },
    { value: 20000, label: '20 km' },
  ];

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
      setCurrentLocation({ lat: latitude, lng: longitude });
      loadTrafficIncidents(latitude, longitude);
    } catch (error) {
      console.error('Location error:', error);
      Alert.alert('Error', 'Unable to get current location');
    }
  };

  const loadTrafficIncidents = async (latitude, longitude) => {
    setLoading(true);
    try {
      const response = await HereApiService.getTrafficIncidents(
        latitude,
        longitude,
        selectedRadius
      );
      
      if (response.success && response.data.TRAFFICITEMS) {
        const incidents = response.data.TRAFFICITEMS.TRAFFICITEM || [];
        setTrafficIncidents(Array.isArray(incidents) ? incidents : [incidents]);
      } else {
        setTrafficIncidents([]);
      }
    } catch (error) {
      Alert.alert('Error', error.message);
      setTrafficIncidents([]);
    } finally {
      setLoading(false);
    }
  };

  const onRefresh = async () => {
    setRefreshing(true);
    if (currentLocation) {
      await loadTrafficIncidents(currentLocation.lat, currentLocation.lng);
    } else {
      await getCurrentLocation();
    }
    setRefreshing(false);
  };

  const handleRadiusChange = (radius) => {
    setSelectedRadius(radius);
    if (currentLocation) {
      loadTrafficIncidents(currentLocation.lat, currentLocation.lng);
    }
  };

  const getIncidentIcon = (incident) => {
    if (incident.TRAFFICITEMDETAIL.ROADCLOSED) return '🚫';
    if (incident.CRITICALITY === 'critical') return '🔴';
    if (incident.CRITICALITY === 'major') return '🟡';
    return '🟢';
  };

  const getIncidentSeverity = (incident) => {
    if (incident.TRAFFICITEMDETAIL.ROADCLOSED) return 'Road Closed';
    return incident.CRITICALITY ? incident.CRITICALITY.toUpperCase() : 'MINOR';
  };

  const formatDistance = (incident) => {
    if (!currentLocation || !incident.LOCATION?.GEOLOC?.ORIGIN) return 'N/A';
    
    const lat1 = currentLocation.lat;
    const lon1 = currentLocation.lng;
    const lat2 = parseFloat(incident.LOCATION.GEOLOC.ORIGIN.LATITUDE);
    const lon2 = parseFloat(incident.LOCATION.GEOLOC.ORIGIN.LONGITUDE);
    
    const R = 6371; // Earth's radius in km
    const dLat = (lat2 - lat1) * Math.PI / 180;
    const dLon = (lon2 - lon1) * Math.PI / 180;
    const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
              Math.cos(lat1 * Math.PI/180) * Math.cos(lat2 * Math.PI/180) *
              Math.sin(dLon/2) * Math.sin(dLon/2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
    const distance = R * c;
    
    if (distance < 1) {
      return `${Math.round(distance * 1000)} m`;
    }
    return `${distance.toFixed(1)} km`;
  };

  const TrafficIncidentCard = ({ incident, index }) => (
    <View style={styles.incidentCard}>
      <View style={styles.incidentHeader}>
        <View style={styles.incidentIconContainer}>
          <Text style={styles.incidentIcon}>{getIncidentIcon(incident)}</Text>
          <Text style={styles.incidentSeverity}>
            {getIncidentSeverity(incident)}
          </Text>
        </View>
        <Text style={styles.incidentDistance}>
          {formatDistance(incident)} away
        </Text>
      </View>
      
      <Text style={styles.incidentDescription}>
        {incident.TRAFFICITEMDETAIL.DESCRIPTION || 'Traffic incident reported'}
      </Text>
      
      {incident.LOCATION?.DESCRIPTION && (
        <Text style={styles.incidentLocation}>
          📍 {incident.LOCATION.DESCRIPTION}
        </Text>
      )}
      
      <View style={styles.incidentDetails}>
        {incident.STARTTIME && (
          <Text style={styles.incidentTime}>
            Started: {new Date(incident.STARTTIME).toLocaleString()}
          </Text>
        )}
        
        {incident.ENDTIME && (
          <Text style={styles.incidentTime}>
            Expected end: {new Date(incident.ENDTIME).toLocaleString()}
          </Text>
        )}
        
        {incident.TRAFFICITEMDETAIL.ROADCLOSED && (
          <Text style={styles.roadClosed}>⚠ Road is closed</Text>
        )}
      </View>
    </View>
  );

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Traffic Information</Text>
        <Text style={styles.subtitle}>
          {trafficIncidents.length} incidents in your area
        </Text>
      </View>

      {/* Location Status */}
      <View style={styles.section}>
        <View style={styles.locationStatus}>
          <Text style={styles.locationStatusText}>
            📍 {currentLocation 
              ? `Location: ${currentLocation.lat.toFixed(4)}, ${currentLocation.lng.toFixed(4)}`
              : 'Getting your location...'
            }
          </Text>
          {!currentLocation && (
            <TouchableOpacity
              style={styles.refreshLocationButton}
              onPress={getCurrentLocation}
            >
              <Text style={styles.refreshLocationText}>Refresh</Text>
            </TouchableOpacity>
          )}
        </View>
      </View>

      {/* Radius Selection */}
      <View style={styles.radiusContainer}>
        <Text style={styles.radiusLabel}>Search radius:</Text>
        <ScrollView horizontal showsHorizontalScrollIndicator={false}>
          {radiusOptions.map((option) => (
            <TouchableOpacity
              key={option.value}
              style={[
                styles.radiusButton,
                selectedRadius === option.value && styles.selectedRadiusButton
              ]}
              onPress={() => handleRadiusChange(option.value)}
            >
              <Text
                style={[
                  styles.radiusButtonText,
                  selectedRadius === option.value && styles.selectedRadiusText
                ]}
              >
                {option.label}
              </Text>
            </TouchableOpacity>
          ))}
        </ScrollView>
      </View>

      {/* Refresh Button */}
      <TouchableOpacity
        style={styles.refreshButton}
        onPress={onRefresh}
        disabled={loading}
      >
        {loading ? (
          <ActivityIndicator color="white" />
        ) : (
          <Text style={styles.refreshButtonText}>🔄 Refresh Traffic Data</Text>
        )}
      </TouchableOpacity>

      {/* Traffic Incidents List */}
      <ScrollView
        style={styles.incidentsList}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }
      >
        {loading && trafficIncidents.length === 0 ? (
          <View style={styles.loadingContainer}>
            <ActivityIndicator size="large" color="#007AFF" />
            <Text style={styles.loadingText}>Loading traffic incidents...</Text>
          </View>
        ) : trafficIncidents.length === 0 ? (
          <View style={styles.noIncidentsContainer}>
            <Text style={styles.noIncidentsIcon}>✅</Text>
            <Text style={styles.noIncidentsText}>
              No traffic incidents in your area
            </Text>
            <Text style={styles.noIncidentsSubtext}>
              Traffic conditions look good!
            </Text>
          </View>
        ) : (
          trafficIncidents.map((incident, index) => (
            <TrafficIncidentCard key={index} incident={incident} index={index} />
          ))
        )}
      </ScrollView>
    </View>
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
  subtitle: {
    fontSize: 14,
    color: 'rgba(255, 255, 255, 0.8)',
    textAlign: 'center',
    marginTop: 5,
  },
  section: {
    margin: 10,
    marginBottom: 10,
  },
  locationStatus: {
    backgroundColor: 'white',
    padding: 15,
    borderRadius: 10,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  locationStatusText: {
    fontSize: 14,
    color: '#666',
    flex: 1,
  },
  refreshLocationButton: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 15,
    paddingVertical: 8,
    borderRadius: 5,
  },
  refreshLocationText: {
    color: 'white',
    fontSize: 12,
    fontWeight: 'bold',
  },
  radiusContainer: {
    backgroundColor: 'white',
    padding: 15,
    margin: 10,
    borderRadius: 10,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  radiusLabel: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  radiusButton: {
    paddingHorizontal: 15,
    paddingVertical: 8,
    marginRight: 10,
    borderRadius: 20,
    backgroundColor: '#f0f0f0',
  },
  selectedRadiusButton: {
    backgroundColor: '#007AFF',
  },
  radiusButtonText: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#333',
  },
  selectedRadiusText: {
    color: 'white',
  },
  refreshButton: {
    backgroundColor: '#007AFF',
    margin: 10,
    padding: 15,
    borderRadius: 10,
    alignItems: 'center',
  },
  refreshButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  incidentsList: {
    flex: 1,
    padding: 10,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingTop: 100,
  },
  loadingText: {
    marginTop: 10,
    fontSize: 16,
    color: '#666',
  },
  noIncidentsContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingTop: 100,
  },
  noIncidentsIcon: {
    fontSize: 48,
    marginBottom: 20,
  },
  noIncidentsText: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 10,
  },
  noIncidentsSubtext: {
    fontSize: 14,
    color: '#666',
  },
  incidentCard: {
    backgroundColor: 'white',
    borderRadius: 10,
    padding: 15,
    marginBottom: 10,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  incidentHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  incidentIconContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  incidentIcon: {
    fontSize: 24,
    marginRight: 10,
  },
  incidentSeverity: {
    fontSize: 12,
    fontWeight: 'bold',
    color: '#FF3B30',
    backgroundColor: 'rgba(255, 59, 48, 0.1)',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
  },
  incidentDistance: {
    fontSize: 12,
    color: '#666',
    fontWeight: 'bold',
  },
  incidentDescription: {
    fontSize: 16,
    marginBottom: 10,
    lineHeight: 22,
  },
  incidentLocation: {
    fontSize: 14,
    color: '#666',
    marginBottom: 10,
  },
  incidentDetails: {
    borderTopWidth: 1,
    borderTopColor: '#eee',
    paddingTop: 10,
  },
  incidentTime: {
    fontSize: 12,
    color: '#666',
    marginBottom: 5,
  },
  roadClosed: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#FF3B30',
    marginTop: 5,
  },
});

export default TrafficScreen;