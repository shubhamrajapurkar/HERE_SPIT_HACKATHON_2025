import HereMap from '@/components/HereMap'; // update path if needed
import React from 'react';
import { StyleSheet, View } from 'react-native';


export default function HomeScreen() {
  return (
    <View style={styles.container}>
      <HereMap />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
});
