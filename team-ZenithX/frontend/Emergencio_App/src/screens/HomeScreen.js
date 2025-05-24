// // src/screens/HomeScreen.js
// import React from 'react';
// import { View, Text, Button, StyleSheet } from 'react-native';

// const HomeScreen = ({ navigation }) => {
//   return (
//     <View style={styles.container}>
//       <Text style={styles.title}>Emergency Assist App</Text>
//       <Button
//         title="Go to Map"
//         onPress={() => navigation.navigate('Map')}
//       />
//     </View>
//   );
// };

// export default HomeScreen;

// const styles = StyleSheet.create({
//   container: {
//     flex: 1,
//     justifyContent: 'center',
//     alignItems: 'center',
//     padding: 20
//   },
//   title: {
//     fontSize: 24,
//     marginBottom: 20,
//     fontWeight: 'bold',
//   },
// });


import axios from 'axios';
import React, { useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Alert, ScrollView } from 'react-native';
import { API_BASE_URL } from '../services/HereApiService';


const HomeScreen = ({ navigation }) => {
  const [isSOSActive, setIsSOSActive] = useState(false);

  const handleSOSPress = async() => {
    setIsSOSActive(true);
    //actually hit api to send SOS signal
    try {
     await axios.post(API_BASE_URL+"/message", {
        message: 'SOS! I need help!',
        location: {
          latitude: 19.1231,
          longitude: 72.8358,
        },
      })
    } catch (error) {
      console.error('Error sending SOS:', error);
      Alert.alert('Error', 'Failed to send SOS signal. Please try again.');
      setIsSOSActive(false);
      return;
      
    }

    Alert.alert(
      'Emergency Alert', 
      'SOS signal sent to emergency services and your emergency contacts!',
      [
        { text: 'OK', onPress: () => setIsSOSActive(false) }
      ]
    );
  };

  const handleQuickAction = (action) => {
    Alert.alert('Quick Action',` ${action} feature activated`);
  };

  const handleGoToMap = () => {
    navigation.navigate('Map');
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.contentContainer}>
      {/* Header Section */}
      <View style={styles.header}>
        <Text style={styles.appTitle}>Emergency Assistant</Text>
        <Text style={styles.subtitle}>Your safety is our priority</Text>
      </View>

      {/* Status Card */}
      <View style={styles.statusCard}>
        <View style={styles.statusIndicator}>
          <View style={styles.safeIndicator} />
          <Text style={styles.statusText}>You're Safe</Text>
        </View>
        <Text style={styles.statusSubtext}>All systems operational</Text>
      </View>

      {/* Map Navigation Button */}
      <TouchableOpacity style={styles.mapButton} onPress={handleGoToMap}>
        <Text style={styles.mapButtonIcon}>🗺</Text>
        <View style={styles.mapButtonContent}>
          <Text style={styles.mapButtonTitle}>Go to  Map</Text>
          <Text style={styles.mapButtonSubtext}>View nearby  services</Text>
        </View>
        <Text style={styles.mapButtonArrow}>→</Text>
      </TouchableOpacity>

      {/* SOS Button Section */}
      <View style={styles.sosSection}>
        <Text style={styles.sosTitle}>Emergency SOS</Text>
        <Text style={styles.sosDescription}>
          Hold and press the SOS button to send an emergency alert to authorities and your emergency contacts
        </Text>
        
        <TouchableOpacity 
          style={[styles.sosButton, isSOSActive && styles.sosButtonActive]} 
          onPress={handleSOSPress}
          activeOpacity={0.8}
        >
          <Text style={styles.sosText}>SOS</Text>
          <Text style={styles.sosSubtext}>EMERGENCY</Text>
        </TouchableOpacity>
        
        <Text style={styles.sosInstructions}>
          Tap once to send alert • Hold for 3 seconds for automatic call
        </Text>
      </View>

      {/* Quick Actions */}
      <View style={styles.quickActionsContainer}>
        <Text style={styles.sectionTitle}>Quick Actions</Text>
        <View style={styles.quickActionsGrid}>
          <TouchableOpacity 
            style={styles.quickActionButton}
            onPress={() => handleQuickAction('Medical Emergency')}
          >
            <Text style={styles.quickActionIcon}>🏥</Text>
            <Text style={styles.quickActionText}>Medical</Text>
          </TouchableOpacity>
          
          <TouchableOpacity 
            style={styles.quickActionButton}
            onPress={() => handleQuickAction('Fire Emergency')}
          >
            <Text style={styles.quickActionIcon}>🚒</Text>
            <Text style={styles.quickActionText}>Fire</Text>
          </TouchableOpacity>
          
          <TouchableOpacity 
            style={styles.quickActionButton}
            onPress={() => handleQuickAction('Police')}
          >
            <Text style={styles.quickActionIcon}>👮</Text>
            <Text style={styles.quickActionText}>Police</Text>
          </TouchableOpacity>
          
          <TouchableOpacity 
            style={styles.quickActionButton}
            onPress={() => handleQuickAction('Share Location')}
          >
            <Text style={styles.quickActionIcon}>📍</Text>
            <Text style={styles.quickActionText}>Location</Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* Emergency Contacts */}
      <View style={styles.contactsContainer}>
        <Text style={styles.sectionTitle}>Emergency Contacts</Text>
        <View style={styles.contactItem}>
          <Text style={styles.contactName}>Primary Contact</Text>
          <Text style={styles.contactNumber}>+1 (555) 123-4567</Text>
        </View>
        <TouchableOpacity style={styles.addContactButton}>
          <Text style={styles.addContactText}>+ Add Emergency Contact</Text>
        </TouchableOpacity>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#ffffff',
  },
  contentContainer: {
    paddingHorizontal: 20,
    paddingTop: 50,
    paddingBottom: 30,
  },
  
  // Header Styles
  header: {
    alignItems: 'center',
    marginBottom: 30,
  },
  appTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#1a1a1a',
    marginBottom: 5,
  },
  subtitle: {
    fontSize: 16,
    color: '#666666',
    fontWeight: '400',
  },

  // Status Card Styles
  statusCard: {
    backgroundColor: '#f8f9fa',
    borderRadius: 16,
    padding: 20,
    marginBottom: 20,
    borderLeftWidth: 4,
    borderLeftColor: '#28a745',
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 3.84,
    elevation: 5,
  },
  statusIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 5,
  },
  safeIndicator: {
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: '#28a745',
    marginRight: 10,
  },
  statusText: {
    fontSize: 18,
    fontWeight: '600',
    color: '#1a1a1a',
  },
  statusSubtext: {
    fontSize: 14,
    color: '#666666',
    marginLeft: 22,
  },

  // Map Button Styles
  mapButton: {
    backgroundColor: '#007bff',
    borderRadius: 16,
    padding: 20,
    marginBottom: 25,
    flexDirection: 'row',
    alignItems: 'center',
    shadowColor: '#007bff',
    shadowOffset: {
      width: 0,
      height: 3,
    },
    shadowOpacity: 0.2,
    shadowRadius: 5,
    elevation: 6,
  },
  mapButtonIcon: {
    fontSize: 24,
    marginRight: 15,
  },
  mapButtonContent: {
    flex: 1,
  },
  mapButtonTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#ffffff',
    marginBottom: 2,
  },
  mapButtonSubtext: {
    fontSize: 14,
    color: '#e3f2fd',
  },
  mapButtonArrow: {
    fontSize: 20,
    color: '#ffffff',
    fontWeight: 'bold',
  },

  // Quick Actions Styles
  quickActionsContainer: {
    marginBottom: 30,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: '600',
    color: '#1a1a1a',
    marginBottom: 15,
  },
  quickActionsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  quickActionButton: {
    width: '48%',
    backgroundColor: '#f8f9fa',
    borderRadius: 12,
    padding: 20,
    alignItems: 'center',
    marginBottom: 10,
    borderWidth: 1,
    borderColor: '#e9ecef',
  },
  quickActionIcon: {
    fontSize: 24,
    marginBottom: 8,
  },
  quickActionText: {
    fontSize: 14,
    fontWeight: '500',
    color: '#495057',
  },

  // Contacts Styles
  contactsContainer: {
    marginBottom: 40,
  },
  contactItem: {
    backgroundColor: '#f8f9fa',
    borderRadius: 12,
    padding: 15,
    marginBottom: 10,
    borderLeftWidth: 3,
    borderLeftColor: '#007bff',
  },
  contactName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1a1a1a',
    marginBottom: 2,
  },
  contactNumber: {
    fontSize: 14,
    color: '#666666',
  },
  addContactButton: {
    borderWidth: 2,
    borderColor: '#007bff',
    borderStyle: 'dashed',
    borderRadius: 12,
    padding: 15,
    alignItems: 'center',
  },
  addContactText: {
    fontSize: 14,
    color: '#007bff',
    fontWeight: '500',
  },

  // SOS Section Styles
  sosSection: {
    alignItems: 'center',
    paddingTop: 10,
    marginBottom: 35,
  },
  sosTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#dc3545',
    marginBottom: 10,
  },
  sosDescription: {
    fontSize: 14,
    color: '#666666',
    textAlign: 'center',
    marginBottom: 30,
    lineHeight: 20,
    paddingHorizontal: 10,
  },
  sosButton: {
    backgroundColor: '#dc3545',
    width: 180,
    height: 180,
    borderRadius: 90,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 20,
    shadowColor: '#dc3545',
    shadowOffset: {
      width: 0,
      height: 4,
    },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 10,
    borderWidth: 6,
    borderColor: '#ffffff',
  },
  sosButtonActive: {
    backgroundColor: '#b02a37',
    transform: [{ scale: 0.95 }],
  },
  sosText: {
    fontSize: 36,
    color: '#ffffff',
    fontWeight: 'bold',
    letterSpacing: 2,
  },
  sosSubtext: {
    fontSize: 12,
    color: '#ffffff',
    fontWeight: '500',
    marginTop: 4,
    letterSpacing: 1,
  },
  sosInstructions: {
    fontSize: 12,
    color: '#666666',
    textAlign: 'center',
    lineHeight: 16,
  },
});

export default HomeScreen;