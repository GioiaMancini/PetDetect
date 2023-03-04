package com.example.PetDetect;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.MotionEvent;
import android.view.View;
import android.view.animation.Animation;
import android.view.animation.ScaleAnimation;
import android.view.animation.TranslateAnimation;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.example.PetDetect.ml.MobilenetOmbrawilly;
import com.example.PetDetect.ml.MobilenetOmbrawillyQuant;
import com.example.PetDetect.ml.MobilenetV110224Quant;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class MainActivity extends AppCompatActivity {

    Button select_button, capture_button, predict_button, predict_cat_button;
    TextView result_text;
    ImageView image_view;
    Bitmap bitmap;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        getPermission();
        final int REQUEST_IMAGE_CAPTURE = 12;
        String[] labels=new String[1001];
        String[] cat_labels = new String[2];
        int cnt=0;
        try {
            BufferedReader bufferedReader= new BufferedReader(new InputStreamReader(getAssets().open("labels.txt")));
            String line =bufferedReader.readLine();
            while (line!=null){
                labels[cnt]=line;
                cnt++;
                line=bufferedReader.readLine();
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }


        select_button=findViewById(R.id.select_button);
        capture_button=findViewById(R.id.capture_button);
        predict_button=findViewById(R.id.predict_button);
        predict_cat_button=findViewById(R.id.predict_cat_button);
        result_text=findViewById(R.id.result_text);
        image_view=findViewById(R.id.image_view);

        select_button.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View view, MotionEvent motionEvent) {
                switch (motionEvent.getAction()) {
                    case MotionEvent.ACTION_DOWN:
                        Animation anim = new ScaleAnimation(1f, 0.8f, 1f, 0.8f, Animation.RELATIVE_TO_SELF, 0.5f, Animation.RELATIVE_TO_SELF, 0.5f);
                        anim.setDuration(100);
                        anim.setFillAfter(true);
                        view.startAnimation(anim);
                        break;
                    case MotionEvent.ACTION_UP:
                        Animation anim2 = new TranslateAnimation(0f, 0f, 0f, -10f);
                        anim2.setDuration(100);
                        anim2.setFillAfter(true);
                        view.startAnimation(anim2);
                        break;
                    default:
                        break;
                }
                return false;
            }
        });
        select_button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                Intent intent= new Intent();
                intent.setAction(Intent.ACTION_GET_CONTENT);
                intent.setType("image/*");
                startActivityForResult( intent, 10);
            }
        });


        capture_button.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View view, MotionEvent motionEvent) {
                switch (motionEvent.getAction()) {
                    case MotionEvent.ACTION_DOWN:
                        Animation anim = new ScaleAnimation(1f, 0.8f, 1f, 0.8f, Animation.RELATIVE_TO_SELF, 0.5f, Animation.RELATIVE_TO_SELF, 0.5f);
                        anim.setDuration(100);
                        anim.setFillAfter(true);
                        view.startAnimation(anim);
                        break;
                    case MotionEvent.ACTION_UP:
                        Animation anim2 = new TranslateAnimation(0f, 0f, 0f, -10f);
                        anim2.setDuration(100);
                        anim2.setFillAfter(true);
                        view.startAnimation(anim2);
                        break;
                    default:
                        break;
                }
                return false;
            }
        });

        capture_button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);

                startActivityForResult(takePictureIntent, 12);

            }
        });





        predict_button.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View view, MotionEvent motionEvent) {
                switch (motionEvent.getAction()) {
                    case MotionEvent.ACTION_DOWN:
                        Animation anim = new ScaleAnimation(1f, 0.8f, 1f, 0.8f, Animation.RELATIVE_TO_SELF, 0.5f, Animation.RELATIVE_TO_SELF, 0.5f);
                        anim.setDuration(100);
                        anim.setFillAfter(true);
                        view.startAnimation(anim);
                        break;
                    case MotionEvent.ACTION_UP:
                        Animation anim2 = new TranslateAnimation(0f, 0f, 0f, -10f);
                        anim2.setDuration(100);
                        anim2.setFillAfter(true);
                        view.startAnimation(anim2);
                        break;
                    default:
                        break;
                }
                return false;
            }
        });
        predict_button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
            // try
                try {
                    MobilenetV110224Quant model = MobilenetV110224Quant.newInstance(MainActivity.this);

                    // Creates inputs for reference.
                    TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.UINT8);

                    bitmap=Bitmap.createScaledBitmap(bitmap,224,224,true);
                    inputFeature0.loadBuffer(TensorImage.fromBitmap(bitmap).getBuffer());

                    // Runs model inference and gets result.
                    MobilenetV110224Quant.Outputs outputs = model.process(inputFeature0);
                    TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

                    result_text.setText(labels[getMax(outputFeature0.getFloatArray())]+"");


                    // Releases model resources if no longer used.
                    model.close();
                } catch (IOException e) {
                    // TODO Handle the exception
                }
            }
        });


        predict_cat_button.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View view, MotionEvent motionEvent) {
                switch (motionEvent.getAction()) {
                    case MotionEvent.ACTION_DOWN:
                        Animation anim = new ScaleAnimation(1f, 0.8f, 1f, 0.8f, Animation.RELATIVE_TO_SELF, 0.5f, Animation.RELATIVE_TO_SELF, 0.5f);
                        anim.setDuration(100);
                        anim.setFillAfter(true);
                        view.startAnimation(anim);
                        break;
                    case MotionEvent.ACTION_UP:
                        Animation anim2 = new TranslateAnimation(0f, 0f, 0f, -10f);
                        anim2.setDuration(100);
                        anim2.setFillAfter(true);
                        view.startAnimation(anim2);
                        break;
                    default:
                        break;
                }
                return false;
            }
        });
        predict_cat_button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                try {

                    MobilenetOmbrawillyQuant model = MobilenetOmbrawillyQuant.newInstance(MainActivity.this);

                    // Creates inputs for reference.
                    TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.UINT8);

                    bitmap=Bitmap.createScaledBitmap(bitmap,224,224,true);
                    inputFeature0.loadBuffer(TensorImage.fromBitmap(bitmap).getBuffer());

                    // Runs model inference and gets result.
                    MobilenetOmbrawillyQuant.Outputs outputs = model.process(inputFeature0);

                    TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

                    cat_labels[0] ="Ombra";
                    cat_labels[1]="Willy";
                    result_text.setText(cat_labels[getMax(outputFeature0.getFloatArray())]+"");


                    // Releases model resources if no longer used.
                    model.close();
                } catch (IOException e) {
                    // TODO Handle the exception
                }
            }
        });


    }

    int getMax(float[] arr){
        int max=0;
        for(int i=0;i<arr.length;i++){
            if(arr[i]>arr[max]) max=i;
        }
        return max;
    }
    void getPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (checkSelfPermission(Manifest.permission.CAMERA)!= PackageManager.PERMISSION_GRANTED){
                ActivityCompat.requestPermissions(MainActivity.this, new String[]{Manifest.permission.CAMERA},11);
            }
        }

    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {


        if (requestCode == 10) {
            if (data != null) {
                Uri uri = data.getData();
                try {
                    bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
                    image_view.setImageBitmap(bitmap);
                } catch (IOException e) {
                    e.printStackTrace();
                }

            }
        }
        else if(requestCode==12 ) {

               if (resultCode == RESULT_OK) {
                    // La foto è stata scattata correttamente
                   bitmap=(Bitmap) data.getExtras().get("data");
                   image_view.setImageBitmap(bitmap);
                } else if (resultCode == RESULT_CANCELED) {
                    // L'utente è tornato indietro senza scattare la foto
                    //Toast.makeText(this, "Image not taken", Toast.LENGTH_SHORT).show();
                }

        }
        super.onActivityResult(requestCode, resultCode, data);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if(requestCode==11){
            if(grantResults.length>0){
                if (grantResults[0]!=PackageManager.PERMISSION_GRANTED){
                    this.getPermission();
                }
            }
        }
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }
}