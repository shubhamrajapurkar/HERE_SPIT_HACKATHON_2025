import React from 'react';
import { View, Text, StyleSheet } from 'react-native';

export default function EnableJourneyScreen() {
  return (
    <View style={styles.container}>
      <Text style={styles.text}>Enable Journey Screen</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  text: { fontSize: 24 }
});
