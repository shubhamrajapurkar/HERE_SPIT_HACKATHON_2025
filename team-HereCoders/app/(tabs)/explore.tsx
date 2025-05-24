import { Ionicons } from '@expo/vector-icons';
import AsyncStorage from '@react-native-async-storage/async-storage';
import * as ImagePicker from 'expo-image-picker';
import React, { useEffect, useState } from 'react';
import {
  Alert,
  FlatList,
  Image,
  Share,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { WebView } from 'react-native-webview';

type Activity = {
  id: number;
  distance: number; // in meters
  elapsed: number; // in seconds
  pace: number; // min/km
  path: { lat: number; lng: number }[];
  timestamp: string;
};

const HERE_API_KEY = 'WZwgbIu9MjlypvZgiMPPLpVX8f6aTnl9gqJEHN1TZaU';

export default function ProfileScreen() {
  const [activities, setActivities] = useState<Activity[]>([]);
  const [profileName, setProfileName] = useState<string>('Atharva Kajwe');
  const [profilePicture, setProfilePicture] = useState<string | null>(null);
  const [editName, setEditName] = useState<boolean>(false);

  useEffect(() => {
    loadProfileData();
    loadActivities();
  }, []);

  const loadProfileData = async () => {
    const storedName = await AsyncStorage.getItem('profileName');
    const storedPicture = await AsyncStorage.getItem('profilePicture');
    if (storedName) setProfileName(storedName);
    if (storedPicture) setProfilePicture(storedPicture);
  };

  const loadActivities = async () => {
    const storedActivities = await AsyncStorage.getItem('activities');
    if (storedActivities) {
      setActivities(JSON.parse(storedActivities) as Activity[]);
    }
  };

  const saveProfileData = async () => {
    await AsyncStorage.setItem('profileName', profileName);
    if (profilePicture) {
      await AsyncStorage.setItem('profilePicture', profilePicture);
    }
  };

  const pickImage = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [1, 1],
      quality: 1,
    });

    if (!result.canceled) {
      setProfilePicture(result.assets[0].uri);
      await AsyncStorage.setItem('profilePicture', result.assets[0].uri);
    }
  };

  const confirmDelete = (id: number) => {
    Alert.alert('Delete Activity', 'Are you sure you want to delete this activity?', [
      { text: 'Cancel', style: 'cancel' },
      { text: 'Delete', onPress: () => deleteActivity(id), style: 'destructive' },
    ]);
  };

  const deleteActivity = async (id: number) => {
    const filteredActivities = activities.filter((activity) => activity.id !== id);
    setActivities(filteredActivities);
    await AsyncStorage.setItem('activities', JSON.stringify(filteredActivities));
    Alert.alert('Deleted', 'The activity has been deleted.');
  };

  // Format pace from min/km float to mm:ss string like "5:30"
  const formatPace = (pace: number) => {
    const minutes = Math.floor(pace);
    const seconds = Math.round((pace - minutes) * 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')} min/km`;
  };

  // Format elapsed seconds to H:MM:SS or MM:SS
  const formatElapsed = (elapsed: number) => {
    const hrs = Math.floor(elapsed / 3600);
    const mins = Math.floor((elapsed % 3600) / 60);
    const secs = elapsed % 60;
    if (hrs > 0) {
      return `${hrs}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const shareActivity = async (item: Activity) => {
    const distanceKm = (item.distance / 1000).toFixed(2);
    const paceStr = formatPace(item.pace);
    const timeStr = formatElapsed(item.elapsed);
    const dateStr = new Date(item.timestamp).toLocaleDateString(undefined, {
      weekday: 'short',
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });

    const message = `My Run - ${distanceKm} km\nPace: ${paceStr}\nTime: ${timeStr}\nDate: ${dateStr}\n#MyRunApp`;

    try {
      await Share.share({
        message,
      });
    } catch (error) {
      alert('Error sharing activity');
    }
  };

  const renderActivity = ({ item }: { item: Activity }) => {
  const html = `<!DOCTYPE html><html><head>
    <meta name="viewport" content="initial-scale=1.0, width=device-width" />
    <style>html, body, #mapContainer { margin: 0; height: 100%; width: 100%; }</style>
    <script src="https://js.api.here.com/v3/3.1/mapsjs-core.js"></script>
    <script src="https://js.api.here.com/v3/3.1/mapsjs-service.js"></script>
    <script src="https://js.api.here.com/v3/3.1/mapsjs-ui.js"></script>
    <script src="https://js.api.here.com/v3/3.1/mapsjs-mapevents.js"></script>
    <link rel="stylesheet" href="https://js.api.here.com/v3/3.1/mapsjs-ui.css" />
    </head>
    <body><div id="mapContainer"></div>
    <script>
      let platform = new H.service.Platform({apikey: '${HERE_API_KEY}'});
      let defaultLayers = platform.createDefaultLayers();
      let map = new H.Map(document.getElementById('mapContainer'), defaultLayers.raster.normal.map, {
        zoom: 16,
        center: { lat: ${item.path[0]?.lat}, lng: ${item.path[0]?.lng} }
      });
      let behavior = new H.mapevents.Behavior(new H.mapevents.MapEvents(map));
      let ui = H.ui.UI.createDefault(map, defaultLayers);

      let lineString = new H.geo.LineString();
      ${item.path.map((coord) => `lineString.pushPoint({lat: ${coord.lat}, lng: ${coord.lng}});`).join('')}
      let routePolyline = new H.map.Polyline(lineString, {
        style: { lineWidth: 4, strokeColor: '#2196F3' }
      });
      map.addObject(routePolyline);
      map.getViewModel().setLookAtData({ bounds: routePolyline.getBoundingBox() });
    </script>
    </body></html>`;

  return (
    <View style={styles.activityCard}>
      <Text style={styles.activityDate}>
        {new Date(item.timestamp).toLocaleDateString(undefined, {
          weekday: 'short',
          year: 'numeric',
          month: 'short',
          day: 'numeric',
        })}
      </Text>

      <View style={styles.activityHeader}>
        <View style={styles.activityStat}>
          <Text style={styles.statLabel}>DISTANCE</Text>
          <Text style={styles.statValue}>{(item.distance / 1000).toFixed(2)} km</Text>
        </View>
        <View style={styles.activityStat}>
          <Text style={styles.statLabel}>PACE</Text>
          <Text style={styles.statValue}>{formatPace(item.pace)}</Text>
        </View>
        <View style={styles.activityStat}>
          <Text style={styles.statLabel}>TIME</Text>
          <Text style={styles.statValue}>{formatElapsed(item.elapsed)}</Text>
        </View>
      </View>

      <WebView
        originWhitelist={['*']}
        source={{ html }}
        javaScriptEnabled
        domStorageEnabled
        style={styles.map}
      />

      <View style={styles.buttonRow}>
        <TouchableOpacity
          onPress={() => shareActivity(item)}
          style={[styles.actionButton, { backgroundColor: '#4CAF50' }]}
        >
          <Ionicons name="share-social" size={20} color="white" />
          <Text style={styles.buttonText}>Share</Text>
        </TouchableOpacity>

        <TouchableOpacity
          onPress={() => confirmDelete(item.id)}
          style={[styles.actionButton, { backgroundColor: '#ff4d4d' }]}
        >
          <Ionicons name="trash" size={20} color="white" />
          <Text style={styles.buttonText}>Delete</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
};


  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity onPress={pickImage}>
          <Image
            source={
              profilePicture
                ? { uri: profilePicture }
                : {
                    uri:
                      'https://img.freepik.com/premium-vector/avatar-profile-icon-flat-style-male-user-profile-vector-illustration-isolated-background-man-profile-sign-business-concept_157943-38764.jpg',
                  }
            }
            style={styles.profileImage}
          />
        </TouchableOpacity>

        {editName ? (
          <TextInput
            value={profileName}
            onChangeText={setProfileName}
            onBlur={() => {
              setEditName(false);
              saveProfileData();
            }}
            style={styles.profileNameInput}
            autoFocus
          />
        ) : (
          <TouchableOpacity onPress={() => setEditName(true)}>
            <Text style={styles.profileName}>{profileName}</Text>
          </TouchableOpacity>
        )}
      </View>

      <FlatList
        data={activities}
        keyExtractor={(item) => item.id.toString()}
        renderItem={renderActivity}
        contentContainerStyle={styles.activityList}
        ListEmptyComponent={<Text style={styles.noActivitiesText}>No activities recorded yet.</Text>}
      />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  header: {
    alignItems: 'center',
    padding: 16,
    backgroundColor: '#f8f8f8',
  },
  profileImage: {
    width: 150,
    height: 150,
    borderRadius: 75,
  },
  profileName: {
    marginTop: 16,
    fontSize: 24,
    fontWeight: 'bold',
  },
  profileNameInput: {
    marginTop: 16,
    fontSize: 24,
    fontWeight: 'bold',
    borderBottomWidth: 1,
    borderBottomColor: '#ccc',
    textAlign: 'center',
  },
  activityList: {
    paddingHorizontal: 16,
    paddingBottom: 16,
  },
  activityCard: {
    flexDirection: 'column',
    backgroundColor: '#fff',
    padding: 16,
    borderRadius: 12,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOpacity: 0.12,
    shadowOffset: { width: 0, height: 3 },
    shadowRadius: 6,
    elevation: 5,
  },
  activityHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 6,
  },
  activityStat: {
    alignItems: 'center',
    flex: 1,
  },
  statLabel: {
    fontSize: 12,
    color: '#888',
    fontWeight: '600',
  },
  statValue: {
    fontSize: 18,
    fontWeight: '700',
    color: '#222',
    marginTop: 2,
  },
  activityDate: {
    fontSize: 14,
    color: '#555',
    textAlign: 'left',
    marginBottom: 10,
    fontWeight: 500
  },
  map: {
    height: 200,
    borderRadius: 12,
  },
  buttonRow: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
    marginTop: 12,
    gap: 12,
  },
  actionButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderRadius: 25,
    shadowColor: '#000',
    shadowOpacity: 0.15,
    shadowOffset: { width: 0, height: 3 },
    shadowRadius: 6,
    elevation: 4,
  },
  buttonText: {
    color: '#fff',
    fontWeight: '700',
    fontSize: 14,
    marginLeft: 6,
  },
  noActivitiesText: {
    textAlign: 'center',
    marginTop: 16,
    fontSize: 16,
    color: '#888',
  },
});
