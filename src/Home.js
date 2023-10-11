import React, { useState } from "react"
import axios from "axios"

const Home=()=>{
    const [selected,setSelected]=useState("1");
    const [day,setDay]=useState("2023-10-06");

    const handleSelect=(e)=>{
        setSelected(e.target.value);
    }

    const handleDate=(e)=>{
        setDay(e.target.value);
    }

    const [price,setPrice]=useState(0);

    const click=async()=>{
        try{
            await axios.get(`http://localhost:8000/${selected}/${day}`).then((res)=>{setPrice(res.data);})
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
                        <option value="2">하이닉스</option>
                        <option value="3">현대차</option>
                        <option value="4">네이버</option>
                    </select>
                </div>
                <div className="day">
                    <label >날짜를 선택해주세요</label>
                    <br/>
                    <br/>
                    <input onChange={handleDate} value={day} type="date" min="2018-01-01" max="2023-10-06">
                    </input>
                </div>
                <button className="button" onClick={click}>확인</button>
            </div>
            <div className="result">
                <p>예측 결과</p>
                <p>{price}</p>
            </div>
        </div>
    )
}

export default Home;