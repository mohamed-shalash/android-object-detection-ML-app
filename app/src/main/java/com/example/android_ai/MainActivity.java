package com.example.android_ai;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.view.View;
import android.widget.*;

import com.example.android_ai.ml.Iris;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;

public class MainActivity extends AppCompatActivity {
    EditText et_1,et_2,et_3,et_4;
    TextView tv;
    Button btn;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        et_1 =findViewById(R.id.main_text_1);
        et_2 =findViewById(R.id.main_text_2);
        et_3 =findViewById(R.id.main_text_3);
        et_4 =findViewById(R.id.main_text_4);
        tv =findViewById(R.id.textView);
        btn =findViewById(R.id.main_btn);

        btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Float f1 = Float.parseFloat(et_1.getText().toString());
                Float f2 = Float.parseFloat(et_2.getText().toString());
                Float f3 = Float.parseFloat(et_3.getText().toString());
                Float f4 = Float.parseFloat(et_4.getText().toString());

                ByteBuffer byteBuffer =ByteBuffer.allocateDirect(4*4);

                byteBuffer.putFloat(f1);
                byteBuffer.putFloat(f2);
                byteBuffer.putFloat(f3);
                byteBuffer.putFloat(f4);

                try {
                    Iris model = Iris.newInstance(getBaseContext());

                    // Creates inputs for reference.
                    TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 4}, DataType.FLOAT32);
                    inputFeature0.loadBuffer(byteBuffer);

                    // Runs model inference and gets result.
                    Iris.Outputs outputs = model.process(inputFeature0);
                    float[] outputFeature0 = outputs.getOutputFeature0AsTensorBuffer().getFloatArray();
                    // Releases model resources if no longer used.
                    tv.setText(outputFeature0[0]+"\n"+outputFeature0[1]+"\n"+outputFeature0[2]+"\n");
                    model.close();
                } catch ( IOException e) {
                    // TODO Handle the exception
                }
            }
        });

    }
}