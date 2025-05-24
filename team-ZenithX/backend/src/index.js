import express from 'express';
import { calculateRoute, getAlternativeRoutes, getcode, getIncidents, getreverse, getTrafficIncidents } from './controllers/route.controller.js';
import { emergencyRoute } from './controllers/emergency.controller.js';
import { sendMessage } from './controllers/twilio.controller.js';
import dotenv from 'dotenv';


const app=express();
const PORT=8000;

dotenv.config();
app.use(express.json());


app.post('/routes/calulate', calculateRoute);
app.get('/routes/traffic', getTrafficIncidents);
app.post('/routes/emergency', emergencyRoute);
app.post('/routes/alternate', getAlternativeRoutes);
app.post('/message', sendMessage);
app.get('/tp', getIncidents);
app.post('/reverse',getreverse);
app.post('/code',getcode);





app.get('/', (req, res) => {
  res.send('Welcome to the HERE API Integration');
});


app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
  });