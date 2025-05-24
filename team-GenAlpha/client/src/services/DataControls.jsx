import React, { useState } from 'react';

const DataControls = ({ onAddLocation, selectedLocation }) => {
  const [formData, setFormData] = useState({
    name: '',
    address: '',
    type: 'shop',
    lat: '',
    lng: ''
  });

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onAddLocation({
      ...formData,
      coordinates: {
        lat: parseFloat(formData.lat),
        lng: parseFloat(formData.lng)
      }
    });
    setFormData({
      name: '',
      address: '',
      type: 'shop',
      lat: '',
      lng: ''
    });
  };

  return (
    <div className="data-controls">
      <h2>Add New Location</h2>
      <form onSubmit={handleSubmit}>
        <div>
          <label>Name:</label>
          <input 
            type="text" 
            name="name" 
            value={formData.name} 
            onChange={handleChange} 
            required 
          />
        </div>
        <div>
          <label>Address:</label>
          <input 
            type="text" 
            name="address" 
            value={formData.address} 
            onChange={handleChange} 
            required 
          />
        </div>
        <div>
          <label>Type:</label>
          <select name="type" value={formData.type} onChange={handleChange}>
            <option value="shop">Shop</option>
            <option value="restaurant">Restaurant</option>
            <option value="service">Service</option>
          </select>
        </div>
        <div>
          <label>Latitude:</label>
          <input 
            type="number" 
            step="any" 
            name="lat" 
            value={formData.lat} 
            onChange={handleChange} 
            required 
          />
        </div>
        <div>
          <label>Longitude:</label>
          <input 
            type="number" 
            step="any" 
            name="lng" 
            value={formData.lng} 
            onChange={handleChange} 
            required 
          />
        </div>
        <button type="submit">Add Location</button>
      </form>

      {selectedLocation && (
        <div className="location-details">
          <h3>{selectedLocation.name}</h3>
          <p>{selectedLocation.address}</p>
          <p>Type: {selectedLocation.type}</p>
          <p>Coordinates: {selectedLocation.lat}, {selectedLocation.lng}</p>
        </div>
      )}
    </div>
  );
};

export default DataControls;