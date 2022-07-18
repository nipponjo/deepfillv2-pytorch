import React from "react";

const ResultItem = (props) => {
  return (
    <div>
      <h2>{props.data.name}</h2>
      {props.data.output.map((returnVal) => {
        return (
          <div key={returnVal.name}>
            <h3>{returnVal.name}</h3>
            <img src={`${returnVal.file}?${props.outputIdx}`} alt={returnVal.name} />
          </div>
        );
      })}
    </div>
  );
};

export default ResultItem;
