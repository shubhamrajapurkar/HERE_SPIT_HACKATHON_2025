import { View, Text, StyleSheet, Image, ScrollView } from 'react-native';
import React, { useState } from 'react';
import { Picker } from '@react-native-picker/picker';

const stations = {
  "CSMT": require('../../../assets/images/csmt.jpg'),
  "Masjid": require('../../../assets/images/masjid.jpg'),
  "Sandhurst Road": require('../../../assets/images/sandhurst road.jpg'),
  "Byculla": require('../../../assets/images/byculla.jpg'),
  "Chinchpokhli": require('../../../assets/images/chinchpokhli.jpg'),
  "Currey Road": require('../../../assets/images/currey road.jpg'),
  "Parel": require('../../../assets/images/parel.jpg'),
  "Dadar Central": require('../../../assets/images/dadar.jpg'),
  "Matunga": require('../../../assets/images/matunga.jpg'),
  "Sion": require('../../../assets/images/sion.jpg'),
  "Kurla": require('../../../assets/images/kurla.jpg'),
  "Vidyavihar": require('../../../assets/images/vidyavihar.jpg'),
  "Ghatkopar": require('../../../assets/images/ghatkopar.jpg'),
  "Vikhroli": require('../../../assets/images/vikhroli.jpg'),
  "Kanjurmarg": require('../../../assets/images/kanjurmarg.jpg'),
  "Bhandup": require('../../../assets/images/bhandup.jpg'),
  "Nahur": require('../../../assets/images/nahur.jpg'),
  "Mulund": require('../../../assets/images/mulund.jpg'),
  "Thane": require('../../../assets/images/thane.jpg'),
};

const Timetable = () => {
  const [selectedStation, setSelectedStation] = useState('CSMT');

  return (
    <ScrollView contentContainerStyle={styles.scrollContainer}>
      <View style={styles.container}>
        <View style={styles.pickerContainer}>
          <Text style={styles.lineTitle}>Central Line</Text>
          <Picker
            selectedValue={selectedStation}
            onValueChange={(itemValue) => setSelectedStation(itemValue)}
            style={styles.picker}
          >
            {Object.keys(stations).map((station) => (
              <Picker.Item key={station} label={station} value={station} />
            ))}
          </Picker>
        </View>

        <View style={styles.imageContainer}>
          <Image 
            source={stations[selectedStation]}
            style={styles.stationImage}
            resizeMode="cover"
          />
        </View>

        <View style={styles.infoContainer}>
          <Text style={styles.stationName}>{selectedStation}</Text>
          <Text style={styles.infoText}>Station Code: {selectedStation.substring(0,3).toUpperCase()}</Text>
          <Text style={styles.lineInfo}>Central Line - Mumbai Division</Text>
        </View>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  scrollContainer: {
    flexGrow: 1,
  },
  container: {
    flex: 1,
    backgroundColor: '#f3f4f6',
    padding: 16,
  },
  pickerContainer: {
    borderWidth: 1,
    borderColor: '#065f46',
    borderRadius: 8,
    marginBottom: 20,
    backgroundColor: '#d1fae5',
    padding: 10,
    shadowColor: '#065f46',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.2,
    shadowRadius: 3,
    elevation: 3,
  },
  lineTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#065f46',
    marginBottom: 8,
    textAlign: 'center',
  },
  picker: {
    height: 60,
  },
  imageContainer: {
    height: 500,
    borderRadius: 12,
    overflow: 'hidden',
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
    elevation: 5,
    borderWidth: 1,
    borderColor: '#065f46',
  },
  stationImage: {
    width: '100%',
    height: '100%',
  },
  infoContainer: {
    padding: 16,
    backgroundColor: '#d1fae5',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#065f46',
    shadowColor: '#065f46',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.2,
    shadowRadius: 3,
    elevation: 3,
  },
  stationName: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 8,
    color: '#065f46',
  },
  infoText: {
    fontSize: 16,
    color: '#4b5563',
    marginBottom: 4,
  },
  lineInfo: {
    fontSize: 14,
    color: '#065f46',
    fontStyle: 'italic',
  },
});

export default Timetable;
