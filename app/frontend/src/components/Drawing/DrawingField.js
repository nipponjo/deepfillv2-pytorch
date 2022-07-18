import React from "react";
import styles from "./DrawingField.module.css";
import Dropzone from "../UI/Dropzone";

const DrawingField = (props) => {
  const scrollWidth = ({ nativeEvent }) => {
    props.setBrushSize((size) => size - Math.sign(nativeEvent.deltaY));
  };

  return (
    <div
      onDragEnter={() => {
        props.setShowDropzone(true);
      }}
      className={styles["drawing-field"]}
    >
      {(!props.selectedImgUrl || props.showDropzone) && (
        <Dropzone
          style={{
            height: props.imgSizeMax,
            width: props.imgSizeMax,
          }}
          onDropFile={props.dropFileHandler}
        />
      )}

      {props.imageSize && props.selectedImgUrl && (
        <img
          className="background"
          src={props.selectedImgUrl}
          alt="img"
          style={{
            height: props.imageSize.height,
            width: props.imageSize.width,
          }}
        />
      )}

      <canvas
        onMouseDown={props.startDrawing}
        onMouseUp={props.finishDrawing}
        onMouseMove={props.draw}
        ref={props.canvasRef}
        height={props.imageSize.height}
        width={props.imageSize.width}
        onWheel={scrollWidth}
        style={{
          height: props.imageSize.height,
          width: props.imageSize.width,
        }}
      />
    </div>
  );
};

export default DrawingField;
