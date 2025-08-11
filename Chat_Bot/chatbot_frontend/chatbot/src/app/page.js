'use client';

import Image from "next/image";
import { useState, useRef, useEffect } from "react";

export default function Home() {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState("");
  const messagesEndRef = useRef(null);

  // Auto-scroll to the bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputMessage.trim()) return;
    
    // Add user message
    const userMessage = {
      text: inputMessage,
      sender: 'user',
      timestamp: new Date().toISOString()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInputMessage("");
    
    try {
      // Send message to backend API
      const response = await fetch('http://127.0.0.1:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: inputMessage })
      });
      
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      
      const data = await response.json();
      
      // Add bot response
      const botResponse = {
        text: data.response || 'Sorry, I could not process your request.',
        sender: 'bot',
        timestamp: new Date().toISOString()
      };
      
      setMessages(prev => [...prev, botResponse]);
    } catch (error) {
      console.error('Error:', error);
      
      // Add error message
      const errorResponse = {
        text: 'Sorry, there was an error processing your request.',
        sender: 'bot',
        timestamp: new Date().toISOString()
      };
      
      setMessages(prev => [...prev, errorResponse]);
    }
  };

  return (
    <div className="h-[100vh] font-mono">
      <div className="flex justify-between h-[10vh] items-center px-6 py-4 shadow-md">
        <div className="text-xl  font-semibold">Tea Support Assistant</div>
        <div className="cursor-pointer">
          <div className="w-6 h-1 bg-black  mb-1 rounded-sm"></div>
          <div className="w-6 h-1 bg-black mb-1 rounded-sm"></div>
          <div className="w-6 h-1 bg-black rounded-sm"></div>
        </div>
      </div>
      <div className="relative h-[90vh] ">
        <Image
          src="/tea_background6.jpg"
          alt="Tea Background"
          fill
          className="object-cover opacity-10 bg-black"
        />
        <div className="absolute inset-0  flex flex-col px-4 overflow-hidden pb-8 pt-2 ">
          
          <div className="flex-1 overflow-y-auto mb-4 space-y-4">
          <div className="flex flex-col items-center mt-4 mb-8">
            <div className="w-40 h-40 rounded-full shadow-lg overflow-hidden flex items-center justify-center bg-white">
              <Image
                src="/tea_background2.jpg"
                alt="Tea Circle"
                width={160}
                height={160}
                className="object-cover opacity-40 w-full h-full"
                priority
              />
            </div>
            <div className="flex flex-col gap-4 mt-6">
              <div className="px-4 py-2 rounded-lg shadow-md text-black text-center  text-md cursor-pointer  bg-gray-100 ">
                fertilize
              </div>
              <div className="px-4 py-2 rounded-lg shadow-md  text-black text-center  text-md cursor-pointer  bg-gray-100 ">
                diseases
              </div>
              <div className="px-4 py-2 rounded-lg shadow-md  text-black text-center  text-md cursor-pointer  bg-gray-100">
                company policies
              </div>
              <div className="px-6 py-2   text-black text-center  text-md cursor-pointer   ">
                We are here to help you. Ask anything about your tea cultive
              </div>
            </div>
          </div>
            
            {messages.map((message, index) => (
              <div 
                key={index} 
                className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div 
                  className={`max-w-[70%] rounded-lg p-3 ${message.sender === 'user' 
                    ? 'bg-emerald-600 text-white rounded-tr-none' 
                    : 'bg-white text-gray-800 rounded-tl-none'}`}
                >
                  {message.text.split('\n').map((line, i) => (
                    <span key={i}>
                      {line}
                      {i !== message.text.split('\n').length - 1 && <br />}
                    </span>
                  ))}
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
        </div>

        <div className="absolute px-2 bottom-2 w-full gap-2 flex items-center justify-center ">
          <input
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
            placeholder="Type your message here..."
            className="flex-1 bg-white p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-emerald-500"
          />
          <button 
            onClick={handleSendMessage}
            className="hover:bg-emerald-500 bg-black text-white px-6 py-3 rounded-lg transition-colors duration-300 flex items-center justify-center"
          >
            <span className="hidden md:block">Send</span>
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-5 w-5 ml-2"
              viewBox="0 0 20 20"
              fill="currentColor"
            >
              <path d="M10.894 2.553a1 1 0 00-1.788 0l-7 14a1 1 0 001.169 1.409l5-1.429A1 1 0 009 15.571V11a1 1 0 112 0v4.571a1 1 0 00.725.962l5 1.428a1 1 0 001.17-1.408l-7-14z" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}
