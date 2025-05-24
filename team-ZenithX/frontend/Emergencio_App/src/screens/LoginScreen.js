import React, { useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, StyleSheet, Alert } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';

const DUMMY_EMAIL = 'test@example.com';
const DUMMY_PASSWORD = 'password';

export default function LoginScreen({ setIsLoggedIn }) {
    
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  const handleLogin = async () => {
    // if (!email || !password) {
    //   Alert.alert('Error', 'Please enter both email and password');
    //   return;
    // }

    console.log('Email:', email, 'Password:', password); // Log email and password

    if (email === DUMMY_EMAIL && password === DUMMY_PASSWORD) {
      try {
        await AsyncStorage.setItem('user', JSON.stringify({ email }));
        setIsLoggedIn(true);
        console.log('Login successful');
      } catch (error) {
        Alert.alert('Error', 'Login failed');
      }
    } else {
      Alert.alert('Invalid Credentials', 'Incorrect email or password');
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Login</Text>
      <TextInput
        style={styles.input}
        placeholder="Email"
        value={email}
        onChangeText={setEmail}
        autoCapitalize="none"
        keyboardType="email-address"
      />
      <TextInput
        style={styles.input}
        placeholder="Password"
        value={password}
        onChangeText={setPassword}
        secureTextEntry
      />
      <TouchableOpacity style={styles.button} onPress={handleLogin}>
        <Text style={styles.buttonText}>Login</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, justifyContent: 'center', padding: 20 },
  title: { fontSize: 30, marginBottom: 30, textAlign: 'center' },
  input: {
    borderWidth: 1,
    borderColor: '#6200ea',
    borderRadius: 10,
    padding: 12,
    marginBottom: 15,
  },
  button: {
    backgroundColor: '#6200ea',
    padding: 15,
    borderRadius: 10,
    alignItems: 'center',
  },
  buttonText: { color: '#fff', fontSize: 16 },
});