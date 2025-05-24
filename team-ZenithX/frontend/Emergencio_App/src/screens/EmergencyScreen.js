import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
  ScrollView,
  Linking,
} from 'react-native';
import * as Location from 'expo-location';
import HereApiService from '../services/HereApiService';

const EmergencyScreen = ({ navigation }) => {
  const [currentLocation, setCurrentLocation] = useState(null);
  const [loading, setLoading] = useState(false);
  const [emergencyResults, setEmergencyResults] = useState(null);

  const emergencyTypes = [
    {
      type: 'hospital',
      label: 'Hospital',
      icon: '🏥',
      color: '#FF3B30',
      description: 'Find nearest hospital or medical facility',
    },
    {
      type: 'police',
      label: 'Police Station',
      icon: '👮',
      color: '#007AFF',
      description: 'Find nearest police station',
    },
    {
      type: 'fire',
      label: 'Fire Station',
      icon: '🚒',
      color: '#FF9500',
      description: 'Find nearest fire station',
    },
  ];

  const emergencyNumbers = [
    { label: 'Emergency (911)', number: '911', color: '#FF3B30' },
    { label: 'Police', number: '911', color: '#007AFF' },
    { label: 'Fire Department', number: '911', color: '#FF9500' },
    { label: 'Medical Emergency', number: '911', color: '#FF3B30' },
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
    } catch (error) {
      console.error('Location error:', error);
      Alert.alert('Error', 'Unable to get current location');
    }
  };

  const findEmergencyService = async (emergencyType) => {
    if (!currentLocation) {
      Alert.alert('Error', 'Current location not available');
      return;
    }

    setLoading(true);
    try {
      const response = await HereApiService.getEmergencyRoute(
        currentLocation,
        emergencyType
      );

      if (response.success) {
        setEmergencyResults({
          type: emergencyType,
          facility: response.facility,
          route: response.route,
        });

        const facility = response.facility;
        const routeSummary = response.route.routes[0].sections[0].summary;
       
        Alert.alert(
          `Nearest ${emergencyType.charAt(0).toUpperCase() + emergencyType.slice(1)} Found`,
          `${facility.title}\n\nDistance: ${(routeSummary.length / 1000).toFixed(1)} km\nEstimated time: ${Math.round(routeSummary.duration / 60)} minutes\n\nAddress: ${facility.address?.label || 'Address not available'}`,
          [
            { text: 'Cancel', style: 'cancel' },
            {
              text: 'Navigate',
              onPress: () => navigation.navigate('Map', {
                emergencyRoute: response,
                emergencyType
              })
            },
            {
              text: 'Call',
              onPress: () => {
                if (facility.contacts && facility.contacts.length > 0) {
                  const phone = facility.contacts[0].phone?.[0]?.value;
                  if (phone) {
                    Linking.openURL(`tel:${phone}`);
                  } else {
                    Alert.alert('No phone number available');
                  }
                } else {
                  Alert.alert('No contact information available');
                }
              }
            }
          ]
        );
      }
    } catch (error) {
      Alert.alert('Error', error.message);
    } finally {
      setLoading(false);
    }
  };

  const callEmergencyNumber = (number) => {
    Alert.alert(
      'Emergency Call',
      `Call ${number}?`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Call Now',
          style: 'destructive',
          onPress: () => Linking.openURL(`tel:${number}`)
        }
      ]
    );
  };

  const formatDistance = (meters) => {
    if (meters >= 1000) {
      return `${(meters / 1000).toFixed(1)} km`;
    }
    return `${meters} m`;
  };

  const formatDuration = (seconds) => {
    const minutes = Math.round(seconds / 60);
    if (minutes >= 60) {
      const hours = Math.floor(minutes / 60);
      const remainingMinutes = minutes % 60;
      return `${hours}h ${remainingMinutes}m`;
    }
    return `${minutes}m`;
  };

  const EmergencyServiceCard = ({ service }) => (
    <TouchableOpacity
      style={[styles.serviceCard, { borderLeftColor: service.color }]}
      onPress={() => findEmergencyService(service.type)}
      disabled={loading}
    >
      <View style={styles.serviceIcon}>
        <Text style={styles.serviceIconText}>{service.icon}</Text>
      </View>
      <View style={styles.serviceInfo}>
        <Text style={styles.serviceLabel}>{service.label}</Text>
        <Text style={styles.serviceDescription}>{service.description}</Text>
      </View>
      <View style={styles.serviceArrow}>
        <Text style={styles.serviceArrowText}>→</Text>
      </View>
    </TouchableOpacity>
  );

  const EmergencyNumberCard = ({ emergency }) => (
    <TouchableOpacity
      style={[styles.emergencyNumberCard, { backgroundColor: emergency.color }]}
      onPress={() => callEmergencyNumber(emergency.number)}
    >
      <Text style={styles.emergencyNumberLabel}>{emergency.label}</Text>
      <Text style={styles.emergencyNumber}>{emergency.number}</Text>
    </TouchableOpacity>
  );

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Emergency Services</Text>
        <Text style={styles.subtitle}>Find help when you need it most</Text>
      </View>

      {/* Emergency Numbers */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>🚨 Emergency Numbers</Text>
        <View style={styles.emergencyNumbersGrid}>
          {emergencyNumbers.map((emergency, index) => (
            <EmergencyNumberCard key={index} emergency={emergency} />
          ))}
        </View>
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

      {/* Emergency Services */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>🏥 Find Emergency Services</Text>
        {emergencyTypes.map((service, index) => (
          <EmergencyServiceCard key={index} service={service} />
        ))}
      </View>

      {/* Recent Emergency Result */}
      {emergencyResults && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>🎯 Last Search Result</Text>
          <View style={styles.resultCard}>
            <View style={styles.resultHeader}>
              <Text style={styles.resultType}>
                {emergencyResults.type.charAt(0).toUpperCase() + emergencyResults.type.slice(1)}
              </Text>
              <Text style={styles.resultName}>
                {emergencyResults.facility.title}
              </Text>
            </View>
           
            {emergencyResults.route.routes && emergencyResults.route.routes.length > 0 && (
              <View style={styles.resultDetails}>
                <Text style={styles.resultDistance}>
                  Distance: {formatDistance(emergencyResults.route.routes[0].sections[0].summary.length)}
                </Text>
                <Text style={styles.resultDuration}>
                  Time: {formatDuration(emergencyResults.route.routes[0].sections[0].summary.duration)}
                </Text>
              </View>
            )}
           
            <View style={styles.resultActions}>
              <TouchableOpacity
                style={styles.resultButton}
                onPress={() => navigation.navigate('Map', {
                  emergencyRoute: emergencyResults,
                  emergencyType: emergencyResults.type
                })}
              >
                <Text style={styles.resultButtonText}>View on Map</Text>
              </TouchableOpacity>
             
              {emergencyResults.facility.contacts && emergencyResults.facility.contacts.length > 0 && (
                <TouchableOpacity
                  style={[styles.resultButton, styles.callButton]}
                  onPress={() => {
                    const phone = emergencyResults.facility.contacts[0].phone?.[0]?.value;
                    if (phone) {
                      callEmergencyNumber(phone);
                    } else {
                      Alert.alert('No phone number available');
                    }
                  }}
                >
                  <Text style={styles.resultButtonText}>Call</Text>
                </TouchableOpacity>
              )}
            </View>
          </View>
        </View>
      )}

      {/* Loading Indicator */}
      {loading && (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#007AFF" />
          <Text style={styles.loadingText}>Finding nearest emergency service...</Text>
        </View>
      )}

      {/* Safety Tips */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>💡 Safety Tips</Text>
        <View style={styles.tipsContainer}>
          <Text style={styles.tip}>• Always call 911 for life-threatening emergencies</Text>
          <Text style={styles.tip}>• Keep your location services enabled</Text>
          <Text style={styles.tip}>• Save important emergency contacts in your phone</Text>
          <Text style={styles.tip}>• Know your address and nearest cross streets</Text>
          <Text style={styles.tip}>• Stay calm and provide clear information to dispatchers</Text>
          <Text style={styles.tip}>• In case of medical emergency, provide patient's age and condition</Text>
          <Text style={styles.tip}>• For fire emergencies, evacuate first, then call</Text>
          <Text style={styles.tip}>• Keep emergency supplies in your car and home</Text>
        </View>
      </View>

      {/* Emergency Preparedness */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>🎒 Emergency Preparedness</Text>
        <View style={styles.preparednessContainer}>
          <View style={styles.preparednessCard}>
            <Text style={styles.preparednessIcon}>🏠</Text>
            <Text style={styles.preparednessTitle}>Home Emergency Kit</Text>
            <Text style={styles.preparednessText}>
              Water (1 gallon per person per day), non-perishable food, flashlight,
              first aid kit, medications, radio, batteries, cash, important documents
            </Text>
          </View>
         
          <View style={styles.preparednessCard}>
            <Text style={styles.preparednessIcon}>🚗</Text>
            <Text style={styles.preparednessTitle}>Car Emergency Kit</Text>
            <Text style={styles.preparednessText}>
              Jumper cables, spare tire, tire iron, jack, emergency flares,
              first aid kit, blanket, water, snacks, phone charger
            </Text>
          </View>
         
          <View style={styles.preparednessCard}>
            <Text style={styles.preparednessIcon}>📱</Text>
            <Text style={styles.preparednessTitle}>Digital Preparedness</Text>
            <Text style={styles.preparednessText}>
              Emergency contact list, medical information, insurance info,
              photos of important documents stored in cloud
            </Text>
          </View>
        </View>
      </View>

      {/* Quick Actions */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>⚡ Quick Actions</Text>
        <View style={styles.quickActionsContainer}>
          <TouchableOpacity
            style={styles.quickActionButton}
            onPress={() => {
              Alert.alert(
                'Share Location',
                'Share your current location with emergency contacts?',
                [
                  { text: 'Cancel', style: 'cancel' },
                  {
                    text: 'Share',
                    onPress: () => {
                      if (currentLocation) {
                        const message = `Emergency! My current location: https://maps.google.com/?q=${currentLocation.lat},${currentLocation.lng}`;
                        Linking.openURL(`sms:?body=${encodeURIComponent(message)}`);
                      } else {
                        Alert.alert('Error', 'Location not available');
                      }
                    }
                  }
                ]
              );
            }}
          >
            <Text style={styles.quickActionIcon}>📍</Text>
            <Text style={styles.quickActionText}>Share Location</Text>
          </TouchableOpacity>
         
          <TouchableOpacity
            style={styles.quickActionButton}
            onPress={() => {
              Alert.alert(
                'Emergency Alert',
                'Send emergency alert to your contacts?',
                [
                  { text: 'Cancel', style: 'cancel' },
                  {
                    text: 'Send Alert',
                    style: 'destructive',
                    onPress: () => {
                      const message = `EMERGENCY ALERT! I need help. My location: ${currentLocation ? `https://maps.google.com/?q=${currentLocation.lat},${currentLocation.lng}` : 'Unknown'}. Please contact me immediately!`;
                      Linking.openURL(`sms:?body=${encodeURIComponent(message)}`);
                    }
                  }
                ]
              );
            }}
          >
            <Text style={styles.quickActionIcon}>🚨</Text>
            <Text style={styles.quickActionText}>Send Alert</Text>
          </TouchableOpacity>
         
          <TouchableOpacity
            style={styles.quickActionButton}
            onPress={() => {
              Alert.alert(
                'Medical Info',
                'Access medical information and emergency contacts',
                [
                  { text: 'OK' }
                ]
              );
            }}
          >
            <Text style={styles.quickActionIcon}>⚕️</Text>
            <Text style={styles.quickActionText}>Medical Info</Text>
          </TouchableOpacity>
        </View>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    backgroundColor: '#FF3B30',
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
    marginBottom: 20,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 15,
    color: '#333',
  },
  emergencyNumbersGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  emergencyNumberCard: {
    width: '48%',
    padding: 20,
    borderRadius: 10,
    alignItems: 'center',
    marginBottom: 10,
  },
  emergencyNumberLabel: {
    color: 'white',
    fontSize: 12,
    fontWeight: 'bold',
    marginBottom: 5,
    textAlign: 'center',
  },
  emergencyNumber: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
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
  serviceCard: {
    backgroundColor: 'white',
    borderRadius: 10,
    padding: 15,
    marginBottom: 10,
    flexDirection: 'row',
    alignItems: 'center',
    borderLeftWidth: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  serviceIcon: {
    marginRight: 15,
  },
  serviceIconText: {
    fontSize: 32,
  },
  serviceInfo: {
    flex: 1,
  },
  serviceLabel: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 5,
  },
  serviceDescription: {
    fontSize: 14,
    color: '#666',
  },
  serviceArrow: {
    marginLeft: 10,
  },
  serviceArrowText: {
    fontSize: 18,
    color: '#999',
  },
  resultCard: {
    backgroundColor: 'white',
    borderRadius: 10,
    padding: 15,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  resultHeader: {
    marginBottom: 10,
  },
  resultType: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#007AFF',
    marginBottom: 5,
  },
  resultName: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
  },
  resultDetails: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 15,
    paddingBottom: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  resultDistance: {
    fontSize: 14,
    color: '#666',
  },
  resultDuration: {
    fontSize: 14,
    color: '#666',
  },
  resultActions: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  resultButton: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 8,
    flex: 0.45,
  },
  callButton: {
    backgroundColor: '#FF3B30',
  },
  resultButtonText: {
    color: 'white',
    textAlign: 'center',
    fontWeight: 'bold',
    fontSize: 14,
  },
  loadingContainer: {
    backgroundColor: 'white',
    margin: 10,
    padding: 30,
    borderRadius: 10,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  loadingText: {
    marginTop: 10,
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
  },
  tipsContainer: {
    backgroundColor: 'white',
    padding: 15,
    borderRadius: 10,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  tip: {
    fontSize: 14,
    color: '#666',
    marginBottom: 8,
    lineHeight: 20,
  },
  preparednessContainer: {
    backgroundColor: 'white',
    borderRadius: 10,
    padding: 15,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  preparednessCard: {
    marginBottom: 20,
    paddingBottom: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  preparednessIcon: {
    fontSize: 32,
    marginBottom: 10,
  },
  preparednessTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 8,
  },
  preparednessText: {
    fontSize: 14,
    color: '#666',
    lineHeight: 20,
  },
  quickActionsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    flexWrap: 'wrap',
  },
  quickActionButton: {
    backgroundColor: 'white',
    width: '31%',
    aspectRatio: 1,
    borderRadius: 10,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 10,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  quickActionIcon: {
    fontSize: 32,
    marginBottom: 8,
  },
  quickActionText: {
    fontSize: 12,
    fontWeight: 'bold',
    color: '#333',
    textAlign: 'center',
  },
});

export default EmergencyScreen;