import React, { useState } from "react"
import axios from "axios"

const Home=()=>{
    const [selected,setSelected]=useState("1");
    const [day,setDay]=useState("2023-10-06");
    const [model,setModel]=useState("1");

    const handleSelect=(e)=>{
        setSelected(e.target.value);
    }

    const handleModel=(e)=>{
        setModel(e.target.value);
    }

    const handleDate=(e)=>{
        setDay(e.target.value);
    }

    const [price,setPrice]=useState(0);
    const [real,setReal]=useState(0);

    const click=async()=>{
        try{
            await axios.get(`http://localhost:8000/${selected}/${day}/${model}`).then((res)=>{setPrice(res.data.price); setReal(res.data.real)})
        }
        catch{
            console.log(selected);
            console.log(day);
        }
    }

    return (
        <div className="main">
            <div className="select">
                <div className="stock">
                    <label>종목을 골라 주세요</label>
                    <br/>
                    <br/>
                    <select onChange={handleSelect} value={selected} className="select">
                        <option value="1">삼성전자</option>
                        <option value="2">현대차</option>
                    </select>
                </div>
                <div className="day">
                    <label >날짜를 선택해주세요 <br/> 선택한 날의 다음 주식 거래일의 결과가 예측됩니다</label>
                    <br/>
                    <br/>
                    <input onChange={handleDate} value={day} type="date" min="2023-01-01" max="2023-10-06">
                    </input>
                </div>
                <div className="model">
                    <label>모델을 골라 주세요</label>
                    <br/>
                    <br/>
                    <select onChange={handleModel} value={model} className="model">
                        <option value="1">LSTM</option>
                        <option value="2">Transformer</option>
                    </select>
                </div>
                <button className="button" onClick={click}>확인</button>
            </div>
            <div className="result">
                <p>예측 결과</p>
                <p>{price}</p>
                <p>실제 주가</p>
                <p>{real}</p>
            </div>
        </div>
    )
}

export default Home;