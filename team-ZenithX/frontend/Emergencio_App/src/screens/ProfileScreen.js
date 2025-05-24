import React, { useState, useContext } from 'react';
import { View, Text, StyleSheet, Image, TouchableOpacity, ScrollView, Alert } from 'react-native';
// import { AppContext } from '../AppContext';
// import * as Contacts from 'expo-contacts';

export default function ProfileScreen() {
  // const { email } = useContext(AppContext);
  const [profileData, setProfileData] = useState({
    name: 'John Doe',
    email: 'john.doe@example.com',
    phone: '+1 (555) 123-4567',
    emergencyContacts: 2,
    profileImage: 'https://via.placeholder.com/120'
  });

  const accessContacts = async () => {
    Alert.alert('Contact Access', 'Emergency contact feature will be implemented here.');
    // console.log('Accessing contacts...');
    // const { status } = await Contacts.requestPermissionsAsync();
    // console.log('Permission status:', status);
    // if (status === 'granted') {
    //   const { data } = await Contacts.getContactsAsync();
    //   console.log('Contacts fetched:', data);
    //   if (data.length > 0) {
    //     alert(Contact: ${data[0].name});
    //   }
    // } else {
    //   alert('Permission to access contacts was denied.');
    // }
  };

  const handleEditProfile = () => {
    Alert.alert('Edit Profile', 'Profile editing feature coming soon!');
  };

  const handleSettingsAction = (action) => {
    Alert.alert('Settings', `${action} feature activated`);
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.contentContainer}>
      {/* Header Section */}
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Profile</Text>
        <TouchableOpacity style={styles.editButton} onPress={handleEditProfile}>
          <Text style={styles.editButtonText}>Edit</Text>
        </TouchableOpacity>
      </View>

      {/* Profile Card */}
      <View style={styles.profileCard}>
        <View style={styles.profileImageContainer}>
          <Image
            source={{ uri: profileData.profileImage }}
            style={styles.profileImage}
          />
          <View style={styles.statusIndicator}>
            <View style={styles.onlineIndicator} />
          </View>
        </View>
        
        <Text style={styles.profileName}>{profileData.name}</Text>
        <Text style={styles.profileEmail}>{profileData.email}</Text>
        <Text style={styles.profilePhone}>{profileData.phone}</Text>
        
        <View style={styles.profileStats}>
          <View style={styles.statItem}>
            <Text style={styles.statNumber}>{profileData.emergencyContacts}</Text>
            <Text style={styles.statLabel}>Emergency Contacts</Text>
          </View>
          <View style={styles.statDivider} />
          <View style={styles.statItem}>
            <Text style={styles.statNumber}>24/7</Text>
            <Text style={styles.statLabel}>Protection</Text>
          </View>
        </View>
      </View>

      {/* Emergency Settings */}
      <View style={styles.settingsSection}>
        <Text style={styles.sectionTitle}>Emergency Settings</Text>
        
        <TouchableOpacity style={styles.settingItem} onPress={accessContacts}>
          <View style={styles.settingContent}>
            <Text style={styles.settingIcon}>👨‍👩‍👧‍👦</Text>
            <View style={styles.settingTextContainer}>
              <Text style={styles.settingTitle}>Emergency Contacts</Text>
              <Text style={styles.settingSubtitle}>Manage your emergency contact list</Text>
            </View>
          </View>
          <Text style={styles.settingArrow}>›</Text>
        </TouchableOpacity>

        <TouchableOpacity 
          style={styles.settingItem} 
          onPress={() => handleSettingsAction('Medical Info')}
        >
          <View style={styles.settingContent}>
            <Text style={styles.settingIcon}>🏥</Text>
            <View style={styles.settingTextContainer}>
              <Text style={styles.settingTitle}>Medical Information</Text>
              <Text style={styles.settingSubtitle}>Blood type, allergies, medications</Text>
            </View>
          </View>
          <Text style={styles.settingArrow}>›</Text>
        </TouchableOpacity>

        <TouchableOpacity 
          style={styles.settingItem} 
          onPress={() => handleSettingsAction('Location')}
        >
          <View style={styles.settingContent}>
            <Text style={styles.settingIcon}>📍</Text>
            <View style={styles.settingTextContainer}>
              <Text style={styles.settingTitle}>Location Services</Text>
              <Text style={styles.settingSubtitle}>Share location in emergencies</Text>
            </View>
          </View>
          <Text style={styles.settingArrow}>›</Text>
        </TouchableOpacity>

        <TouchableOpacity 
          style={styles.settingItem} 
          onPress={() => handleSettingsAction('Notifications')}
        >
          <View style={styles.settingContent}>
            <Text style={styles.settingIcon}>🔔</Text>
            <View style={styles.settingTextContainer}>
              <Text style={styles.settingTitle}>Alert Preferences</Text>
              <Text style={styles.settingSubtitle}>Customize emergency notifications</Text>
            </View>
          </View>
          <Text style={styles.settingArrow}>›</Text>
        </TouchableOpacity>
      </View>

      {/* Safety Features */}
      <View style={styles.safetySection}>
        <Text style={styles.sectionTitle}>Safety Features</Text>
        
        <View style={styles.featureGrid}>
          <TouchableOpacity 
            style={styles.featureItem}
            onPress={() => handleSettingsAction('Auto SOS')}
          >
            <Text style={styles.featureIcon}>⚡</Text>
            <Text style={styles.featureTitle}>Auto SOS</Text>
            <Text style={styles.featureStatus}>Enabled</Text>
          </TouchableOpacity>

          <TouchableOpacity 
            style={styles.featureItem}
            onPress={() => handleSettingsAction('Location Tracking')}
          >
            <Text style={styles.featureIcon}>🛡</Text>
            <Text style={styles.featureTitle}>Safe Mode</Text>
            <Text style={styles.featureStatus}>Active</Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* Action Buttons */}
      <View style={styles.actionButtons}>
        <TouchableOpacity 
          style={styles.primaryButton} 
          onPress={() => handleSettingsAction('Test Emergency')}
        >
          <Text style={styles.primaryButtonText}>Test Emergency System</Text>
        </TouchableOpacity>
        
        <TouchableOpacity 
          style={styles.secondaryButton} 
          onPress={() => handleSettingsAction('Privacy Settings')}
        >
          <Text style={styles.secondaryButtonText}>Privacy & Security</Text>
        </TouchableOpacity>
      </View>
    </ScrollView>
  );
}

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
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 30,
  },
  headerTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#1a1a1a',
  },
  editButton: {
    backgroundColor: '#007bff',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
  },
  editButtonText: {
    color: '#ffffff',
    fontWeight: '600',
    fontSize: 14,
  },

  // Profile Card Styles
  profileCard: {
    backgroundColor: '#f8f9fa',
    borderRadius: 20,
    padding: 30,
    alignItems: 'center',
    marginBottom: 30,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 5,
  },
  profileImageContainer: {
    position: 'relative',
    marginBottom: 20,
  },
  profileImage: {
    width: 120,
    height: 120,
    borderRadius: 60,
    borderWidth: 4,
    borderColor: '#ffffff',
  },
  statusIndicator: {
    position: 'absolute',
    bottom: 5,
    right: 5,
    backgroundColor: '#ffffff',
    borderRadius: 12,
    padding: 2,
  },
  onlineIndicator: {
    width: 16,
    height: 16,
    borderRadius: 8,
    backgroundColor: '#28a745',
  },
  profileName: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#1a1a1a',
    marginBottom: 5,
  },
  profileEmail: {
    fontSize: 16,
    color: '#666666',
    marginBottom: 5,
  },
  profilePhone: {
    fontSize: 16,
    color: '#666666',
    marginBottom: 20,
  },
  profileStats: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  statItem: {
    alignItems: 'center',
    paddingHorizontal: 20,
  },
  statNumber: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#dc3545',
  },
  statLabel: {
    fontSize: 12,
    color: '#666666',
    marginTop: 2,
  },
  statDivider: {
    width: 1,
    height: 30,
    backgroundColor: '#dee2e6',
  },

  // Settings Section Styles
  settingsSection: {
    marginBottom: 30,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: '600',
    color: '#1a1a1a',
    marginBottom: 15,
  },
  settingItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: '#f8f9fa',
    borderRadius: 12,
    padding: 16,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: '#e9ecef',
  },
  settingContent: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  settingIcon: {
    fontSize: 24,
    marginRight: 15,
  },
  settingTextContainer: {
    flex: 1,
  },
  settingTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1a1a1a',
    marginBottom: 2,
  },
  settingSubtitle: {
    fontSize: 14,
    color: '#666666',
  },
  settingArrow: {
    fontSize: 20,
    color: '#666666',
  },

  // Safety Features Styles
  safetySection: {
    marginBottom: 30,
  },
  featureGrid: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  featureItem: {
    width: '48%',
    backgroundColor: '#f8f9fa',
    borderRadius: 12,
    padding: 20,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#e9ecef',
  },
  featureIcon: {
    fontSize: 32,
    marginBottom: 10,
  },
  featureTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1a1a1a',
    marginBottom: 5,
  },
  featureStatus: {
    fontSize: 12,
    color: '#28a745',
    fontWeight: '500',
  },

  // Action Buttons Styles
  actionButtons: {
    gap: 15,
  },
  primaryButton: {
    backgroundColor: '#dc3545',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
  },
  primaryButtonText: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '600',
  },
  secondaryButton: {
    backgroundColor: '#f8f9fa',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#dee2e6',
  },
  secondaryButtonText: {
    color: '#495057',
    fontSize: 16,
    fontWeight: '600',
  },
});