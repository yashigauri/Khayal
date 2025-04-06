import React from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Mic, Heart, Droplets, CloudSun, Thermometer, Gamepad2 } from "lucide-react";

export default function KhayalDashboard() {
  return (
    <div className="p-4 max-w-md mx-auto bg-blue-50 rounded-2xl shadow-xl text-gray-800">
      <div className="text-center">
        <h1 className="text-xl font-bold">GOOD MORNING</h1>
        <p className="text-sm text-gray-600">Meera - Female, 70</p>
      </div>

      <div className="mt-4">
        <input className="w-full p-2 rounded-md border border-gray-300" placeholder="Search..." />
      </div>

      <div className="mt-6">
        <h2 className="font-semibold">Let's Check your daily vitals</h2>
        <div className="grid grid-cols-2 gap-2 mt-2">
          <Card><CardContent className="flex flex-col items-center"><Heart className="text-red-500" /><p>Heart Rate<br/><span className="font-bold">96 bpm</span></p></CardContent></Card>
          <Card><CardContent className="flex flex-col items-center"><Droplets className="text-yellow-600" /><p>Blood Pressure<br/><span className="font-bold">120/72</span></p></CardContent></Card>
          <Card><CardContent className="flex flex-col items-center"><Thermometer className="text-blue-600" /><p>SpO2<br/><span className="font-bold">98</span></p></CardContent></Card>
          <Card><CardContent className="flex flex-col items-center"><CloudSun className="text-blue-400" /><p>Cloudy, 24°C<br/><span className="text-sm">Stay Hydrated 💧</span></p></CardContent></Card>
          <Card className="col-span-2"><CardContent className="flex flex-col items-center"><p>Glucose<br/><span className="font-bold">100</span></p></CardContent></Card>
        </div>
      </div>

      <div className="mt-6">
        <h2 className="font-semibold">RELAX CENTER</h2>
        <div className="grid grid-cols-4 gap-2 mt-2">
          {['Tetris', 'Hungry Snake', 'Sudoku', 'Breath'].map(game => (
            <Card key={game}><CardContent className="text-center text-sm font-medium">{game}</CardContent></Card>
          ))}
        </div>
      </div>

      <div className="mt-6">
        <h2 className="font-semibold">ALERTS</h2>
        <div className="grid grid-cols-3 gap-2 mt-2">
          <Card><CardContent className="text-center text-sm font-medium">Doctors Appointment<br/><span className="font-bold text-sm">2:00 PM</span></CardContent></Card>
          <Card><CardContent className="text-center text-sm font-medium">Medication<br/><span className="font-bold text-sm">8:00 AM</span></CardContent></Card>
          <Card><CardContent className="text-center text-sm font-medium">Exercise<br/><span className="font-bold text-sm">7:00 AM</span></CardContent></Card>
        </div>
      </div>

      <div className="mt-6 flex justify-between items-center">
        <Button className="bg-green-100 text-black">RAKESH</Button>
        <Button variant="ghost" className="text-black"><Mic /></Button>
        <Button className="bg-green-100 text-black">POOJA</Button>
      </div>

      <p className="mt-4 text-center text-yellow-600 font-semibold">Stay positive and keep smiling 😊</p>
    </div>
  );
}
