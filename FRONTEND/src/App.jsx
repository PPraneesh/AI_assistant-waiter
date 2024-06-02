import "./index.css";
import { useForm } from "react-hook-form";
import { useEffect, useState } from "react";
import axios from "axios";

function App() {
  const [messages, setMessages] = useState([]);
const [tableData, setTableData] = useState({});
    useEffect(() => {
      const db = localStorage.getItem("db");
      if (db) {
        setMessages(JSON.parse(db));
      }
      axios.get("https://ai-assistant-waiter-backend.vercel.app/")
        .then((response) => {
        setTableData(response.data.available_tables);
        })
        .catch(() => {
        console.error("hey check the server is running or not");
        });
    }, []);

  const { register, handleSubmit } = useForm();

  const onSubmit = (data,e) => {
    e.preventDefault();
    axios.post("https://ai-assistant-waiter-backend.vercel.app/",data={data})
    .then((response)=>{
      localStorage.setItem("db", JSON.stringify(messages.concat(response.data.messages)));
      setMessages(messages.concat(response.data.messages));
      setTableData(response.data.available_tables);
    })
    .catch((error)=>{
      console.error("An error occurred while submitting the form:", error);
    });
  };
  return (
    <div className="app">
      <h1>WELCOME TO BARATIE</h1>
      <h2>AI ASSISTANT</h2>

      <div className="form">
        <form onSubmit={handleSubmit(onSubmit)}>
          <input type="text" {...register('question')} />
          <button type="submit"><p>go</p></button>
        </form>
      </div>
      <div className="table_data">
        <h2>Available_tables</h2>
        <div className="available_tables">
        {Object.entries(tableData).map(([key, value]) => (
                  <div className="table_time" key={key}>
                  <h3>{key}</h3>
                  <div className="tables" >
                    {
                      Array(value).fill().map((_,index)=>{
                        return <div className="green_square" key={index}>{" "}</div>
                      })
                    }
                    {Array(4-value).fill().map((_,index)=>{
                        return <div className="red_square" key={index}>{" "}</div>
                      })

                    }
                  </div>
                </div>
            ))}
          </div>
      </div>
      <div className="messages">
        <h2>Chat</h2>
        <div className="chat">
        {messages.map((msg,index)=>{
          if(msg.msg_type==="user")
          return <div key={index} className="user_msg">
            <p className="label">user</p>
            <p>{msg.user_question}</p>
            </div>
          else
          return <div key={index} className="ai_msg">
            <p className="label">AI</p>
            <p>{msg.tool_output}</p>
            </div>
        })}
      </div>
      </div>
    </div>
  );
}

export default App;
