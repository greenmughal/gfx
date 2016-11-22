#ifndef NOISE_SIMPLEX_FUNC
#define NOISE_SIMPLEX_FUNC
/*

Description:
	Array- and textureless CgFx/HLSL 2D, 3D and 4D simplex noise functions.
	a.k.a. simplified and optimized Perlin noise.

	The functions have very good performance
	and no dependencies on external data.

	2D - Very fast, very compact code.
	3D - Fast, compact code.
	4D - Reasonably fast, reasonably compact code.

------------------------------------------------------------------

Ported by:
	Lex-DRL
	I've ported the code from GLSL to CgFx/HLSL for Unity,
	added a couple more optimisations (to speed it up even further)
	and slightly reformatted the code to make it more readable.

Original GLSL functions:
	https://github.com/ashima/webgl-noise
	Credits from original glsl file are at the end of this cginc.

------------------------------------------------------------------

Usage:

	float ns = snoise(v);
	// v is any of: float2, float3, float4

	Return type is float.
	To generate 2 or more components of noise (colorful noise),
	call these functions several times with different
	constant offsets for the arguments.
	E.g.:

	float3 colorNs = float3(
		snoise(v),
		snoise(v + 17.0),
		snoise(v - 43.0),
	);


Remark about those offsets from the original author:

	People have different opinions on whether these offsets should be integers
	for the classic noise functions to match the spacing of the zeroes,
	so we have left that for you to decide for yourself.
	For most applications, the exact offsets don't really matter as long
	as they are not too small or too close to the noise lattice period
	(289 in this implementation).

*/

// 1 / 289
#define NOISE_SIMPLEX_1_DIV_289 0.00346020761245674740484429065744f

float mod289(float x) {
	return x - floor(x * NOISE_SIMPLEX_1_DIV_289) * 289.0;
}

float2 mod289(float2 x) {
	return x - floor(x * NOISE_SIMPLEX_1_DIV_289) * 289.0;
}

float3 mod289(float3 x) {
	return x - floor(x * NOISE_SIMPLEX_1_DIV_289) * 289.0;
}

float4 mod289(float4 x) {
	return x - floor(x * NOISE_SIMPLEX_1_DIV_289) * 289.0;
}


// ( x*34.0 + 1.0 )*x =
// x*x*34.0 + x
float permute(float x) {
	return mod289(
		x*x*34.0 + x
	);
}

float3 permute(float3 x) {
	return mod289(
		x*x*34.0 + x
	);
}

float4 permute(float4 x) {
	return mod289(
		x*x*34.0 + x
	);
}



float4 grad4(float j, float4 ip)
{
	const float4 ones = float4(1.0, 1.0, 1.0, -1.0);
	float4 p, s;
	p.xyz = floor( frac(j * ip.xyz) * 7.0) * ip.z - 1.0;
	p.w = 1.5 - dot( abs(p.xyz), ones.xyz );

	// GLSL: lessThan(x, y) = x < y
	// HLSL: 1 - step(y, x) = x < y
	p.xyz -= sign(p.xyz) * (p.w < 0);

	return p;
}



// ----------------------------------- 2D -------------------------------------

float snoise(float2 v)
{
	const float4 C = float4(
		0.211324865405187, // (3.0-sqrt(3.0))/6.0
		0.366025403784439, // 0.5*(sqrt(3.0)-1.0)
	 -0.577350269189626, // -1.0 + 2.0 * C.x
		0.024390243902439  // 1.0 / 41.0
	);

// First corner
	float2 i = floor( v + dot(v, C.yy) );
	float2 x0 = v - i + dot(i, C.xx);

// Other corners
	// float2 i1 = (x0.x > x0.y) ? float2(1.0, 0.0) : float2(0.0, 1.0);
	// Lex-DRL: afaik, step() in GPU is faster than if(), so:
	// step(x, y) = x <= y

	// Actually, a simple conditional without branching is faster than that madness :)
	int2 i1 = (x0.x > x0.y) ? float2(1.0, 0.0) : float2(0.0, 1.0);
	float4 x12 = x0.xyxy + C.xxzz;
	x12.xy -= i1;

// Permutations
	i = mod289(i); // Avoid truncation effects in permutation
	float3 p = permute(
		permute(
				i.y + float3(0.0, i1.y, 1.0 )
		) + i.x + float3(0.0, i1.x, 1.0 )
	);

	float3 m = max(
		0.5 - float3(
			dot(x0, x0),
			dot(x12.xy, x12.xy),
			dot(x12.zw, x12.zw)
		),
		0.0
	);
	m = m*m ;
	m = m*m ;

// Gradients: 41 points uniformly over a line, mapped onto a diamond.
// The ring size 17*17 = 289 is close to a multiple of 41 (41*7 = 287)

	float3 x = 2.0 * frac(p * C.www) - 1.0;
	float3 h = abs(x) - 0.5;
	float3 ox = floor(x + 0.5);
	float3 a0 = x - ox;

// Normalise gradients implicitly by scaling m
// Approximation of: m *= inversesqrt( a0*a0 + h*h );
	m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );

// Compute final noise value at P
	float3 g;
	g.x = a0.x * x0.x + h.x * x0.y;
	g.yz = a0.yz * x12.xz + h.yz * x12.yw;
	return 130.0 * dot(m, g);
}

// ----------------------------------- 3D -------------------------------------

float snoise(float3 v)
{
	const float2 C = float2(
		0.166666666666666667, // 1/6
		0.333333333333333333  // 1/3
	);
	const float4 D = float4(0.0, 0.5, 1.0, 2.0);

// First corner
	float3 i = floor( v + dot(v, C.yyy) );
	float3 x0 = v - i + dot(i, C.xxx);

// Other corners
	float3 g = step(x0.yzx, x0.xyz);
	float3 l = 1 - g;
	float3 i1 = min(g.xyz, l.zxy);
	float3 i2 = max(g.xyz, l.zxy);

	float3 x1 = x0 - i1 + C.xxx;
	float3 x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y
	float3 x3 = x0 - D.yyy;      // -1.0+3.0*C.x = -0.5 = -D.y

// Permutations
	i = mod289(i);
	float4 p = permute(
		permute(
			permute(
					i.z + float4(0.0, i1.z, i2.z, 1.0 )
			) + i.y + float4(0.0, i1.y, i2.y, 1.0 )
		) 	+ i.x + float4(0.0, i1.x, i2.x, 1.0 )
	);

// Gradients: 7x7 points over a square, mapped onto an octahedron.
// The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
	float n_ = 0.142857142857; // 1/7
	float3 ns = n_ * D.wyz - D.xzx;

	float4 j = p - 49.0 * floor(p * ns.z * ns.z); // mod(p,7*7)

	float4 x_ = floor(j * ns.z);
	float4 y_ = floor(j - 7.0 * x_ ); // mod(j,N)

	float4 x = x_ *ns.x + ns.yyyy;
	float4 y = y_ *ns.x + ns.yyyy;
	float4 h = 1.0 - abs(x) - abs(y);

	float4 b0 = float4( x.xy, y.xy );
	float4 b1 = float4( x.zw, y.zw );

	//float4 s0 = float4(lessThan(b0,0.0))*2.0 - 1.0;
	//float4 s1 = float4(lessThan(b1,0.0))*2.0 - 1.0;
	float4 s0 = floor(b0)*2.0 + 1.0;
	float4 s1 = floor(b1)*2.0 + 1.0;
	float4 sh = -step(h, 0.0);

	float4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
	float4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

	float3 p0 = float3(a0.xy,h.x);
	float3 p1 = float3(a0.zw,h.y);
	float3 p2 = float3(a1.xy,h.z);
	float3 p3 = float3(a1.zw,h.w);

//Normalise gradients
	float4 norm = rsqrt(float4(
		dot(p0, p0),
		dot(p1, p1),
		dot(p2, p2),
		dot(p3, p3)
	));
	p0 *= norm.x;
	p1 *= norm.y;
	p2 *= norm.z;
	p3 *= norm.w;

// Mix final noise value
	float4 m = max(
		0.6 - float4(
			dot(x0, x0),
			dot(x1, x1),
			dot(x2, x2),
			dot(x3, x3)
		),
		0.0
	);
	m = m * m;
	return 42.0 * dot(
		m*m,
		float4(
			dot(p0, x0),
			dot(p1, x1),
			dot(p2, x2),
			dot(p3, x3)
		)
	);
}

// ----------------------------------- 4D -------------------------------------

float snoise(float4 v)
{
	const float4 C = float4(
		0.138196601125011, // (5 - sqrt(5))/20 G4
		0.276393202250021, // 2 * G4
		0.414589803375032, // 3 * G4
	 -0.447213595499958  // -1 + 4 * G4
	);

// First corner
	float4 i = floor(
		v +
		dot(
			v,
			0.309016994374947451 // (sqrt(5) - 1) / 4
		)
	);
	float4 x0 = v - i + dot(i, C.xxxx);

// Other corners

// Rank sorting originally contributed by Bill Licea-Kane, AMD (formerly ATI)
	float4 i0;
	float3 isX = step( x0.yzw, x0.xxx );
	float3 isYZ = step( x0.zww, x0.yyz );
	i0.x = isX.x + isX.y + isX.z;
	i0.yzw = 1.0 - isX;
	i0.y += isYZ.x + isYZ.y;
	i0.zw += 1.0 - isYZ.xy;
	i0.z += isYZ.z;
	i0.w += 1.0 - isYZ.z;

	// i0 now contains the unique values 0,1,2,3 in each channel
	float4 i3 = saturate(i0);
	float4 i2 = saturate(i0-1.0);
	float4 i1 = saturate(i0-2.0);

	//	x0 = x0 - 0.0 + 0.0 * C.xxxx
	//	x1 = x0 - i1  + 1.0 * C.xxxx
	//	x2 = x0 - i2  + 2.0 * C.xxxx
	//	x3 = x0 - i3  + 3.0 * C.xxxx
	//	x4 = x0 - 1.0 + 4.0 * C.xxxx
	float4 x1 = x0 - i1 + C.xxxx;
	float4 x2 = x0 - i2 + C.yyyy;
	float4 x3 = x0 - i3 + C.zzzz;
	float4 x4 = x0 + C.wwww;

// Permutations
	i = mod289(i);
	float j0 = permute(
		permute(
			permute(
				permute(i.w) + i.z
			) + i.y
		) + i.x
	);
	float4 j1 = permute(
		permute(
			permute(
				permute (
					i.w + float4(i1.w, i2.w, i3.w, 1.0 )
				) + i.z + float4(i1.z, i2.z, i3.z, 1.0 )
			) + i.y + float4(i1.y, i2.y, i3.y, 1.0 )
		) + i.x + float4(i1.x, i2.x, i3.x, 1.0 )
	);

// Gradients: 7x7x6 points over a cube, mapped onto a 4-cross polytope
// 7*7*6 = 294, which is close to the ring size 17*17 = 289.
	const float4 ip = float4(
		0.003401360544217687075, // 1/294
		0.020408163265306122449, // 1/49
		0.142857142857142857143, // 1/7
		0.0
	);

	float4 p0 = grad4(j0, ip);
	float4 p1 = grad4(j1.x, ip);
	float4 p2 = grad4(j1.y, ip);
	float4 p3 = grad4(j1.z, ip);
	float4 p4 = grad4(j1.w, ip);

// Normalise gradients
	float4 norm = rsqrt(float4(
		dot(p0, p0),
		dot(p1, p1),
		dot(p2, p2),
		dot(p3, p3)
	));
	p0 *= norm.x;
	p1 *= norm.y;
	p2 *= norm.z;
	p3 *= norm.w;
	p4 *= rsqrt( dot(p4, p4) );

// Mix contributions from the five corners
	float3 m0 = max(
		0.6 - float3(
			dot(x0, x0),
			dot(x1, x1),
			dot(x2, x2)
		),
		0.0
	);
	float2 m1 = max(
		0.6 - float2(
			dot(x3, x3),
			dot(x4, x4)
		),
		0.0
	);
	m0 = m0 * m0;
	m1 = m1 * m1;

	return 49.0 * (
		dot(
			m0*m0,
			float3(
				dot(p0, x0),
				dot(p1, x1),
				dot(p2, x2)
			)
		) + dot(
			m1*m1,
			float2(
				dot(p3, x3),
				dot(p4, x4)
			)
		)
	);
}



//                 Credits from source glsl file:
//
// Description : Array and textureless GLSL 2D/3D/4D simplex
//               noise functions.
//      Author : Ian McEwan, Ashima Arts.
//  Maintainer : ijm
//     Lastmod : 20110822 (ijm)
//     License : Copyright (C) 2011 Ashima Arts. All rights reserved.
//               Distributed under the MIT License. See LICENSE file.
//               https://github.com/ashima/webgl-noise
//
//
//           The text from LICENSE file:
//
//
// Copyright (C) 2011 by Ashima Arts (Simplex noise)
// Copyright (C) 2011 by Stefan Gustavson (Classic noise)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
#endif


// fractal sum
float fBm(float3 p, int octaves, float lacunarity = 2.0, float gain = 0.5)
{
	float freq = 1.0, amp = 1.0;
	float sum = 0;
	for(int i=0; i<octaves; i++) {
		sum += snoise(p*freq)*amp;
		freq *= lacunarity;
		amp *= gain;
	}
	return sum;
}

float turbulence(float3 p, int octaves, float lacunarity = 2.0, float gain = 0.5)// 0.5)
{
	float sum = 0;
	float freq = 1.0, amp = 1.0;
	for(int i=0; i<octaves; i++) {
		sum += abs(snoise(p*freq))*amp;
		freq *= lacunarity;
		amp *= gain;
	}
	return sum;
}

float turbulenceZ(float3 p, int octaves, float lacunarity = 2.0, float gain = 0.5)// 0.5)
{
	float sum = 0;
	float freq = 1.0, amp = 1.0;
	for(int i=0; i<octaves; i++) {
		sum += (snoise(p*freq))*amp;
		freq *= lacunarity;
		amp *= gain;
	}
	return min(max(sum, -1.0), 1.0);
}

// Ridged multifractal
// See "Texturing & Modeling, A Procedural Approach", Chapter 12
float ridge(float h, float offset)
{
    h = abs(h);
    h = offset - h;
    h = h * h;
    return h;
}

float ridgedmf(float3 p, int octaves, float lacunarity = 2.0, float gain = 0.5, float offset = 1.0)
{
	// Hmmm... these hardcoded constants make it look nice.  Put on tweakable sliders?
	float f = 0.3 + 0.5 * fBm(p, octaves, lacunarity, gain);
	return ridge(f, offset);
}

// mixture of ridged and fbm noise
float hybridTerrain(float3 x, int3 octaves)
{
	const float SCALE = 256;
	x /= SCALE;

	const int RIDGE_OCTAVES = octaves.x;
	const int FBM_OCTAVES   = octaves.y;
	const int TWIST_OCTAVES = octaves.z;
	const float LACUNARITY = 2, GAIN = 0.5;

	// Distort the ridge texture coords.  Otherwise, you see obvious texel edges.
	//float2 xOffset = float3(fBm(0.2*x, TWIST_OCTAVES), fBm(0.2*x+0.2, TWIST_OCTAVES));
	float3 xTwisted = x + 0.01 ;//* xOffset;

	// Ridged is too ridgy.  So interpolate between ridge and fBm for the coarse octaves.
	float h = ridgedmf(xTwisted, RIDGE_OCTAVES, LACUNARITY, GAIN, 1.0);

	const float fBm_UVScale  = pow(LACUNARITY, RIDGE_OCTAVES);
	const float fBm_AmpScale = pow(GAIN,       RIDGE_OCTAVES);
	float f = fBm(x * fBm_UVScale, FBM_OCTAVES, LACUNARITY, GAIN) * fBm_AmpScale;

	if (RIDGE_OCTAVES > 0)
		return h + f*saturate(h);
	else
		return f;
}


// This allows us to compile the shader with a #define to choose
// the different partition modes for the hull shader.
// See the hull shader: [partitioning(BEZIER_HS_PARTITION)]
// This sample demonstrates "integer", "fractional_even", and "fractional_odd"
#ifndef HS_PARTITION
#define HS_PARTITION "integer"
#endif //HS_PARTITION

// The input patch size.  In this sample, it is 3 vertices(control points).
// This value should match the call to IASetPrimitiveTopology()
#define INPUT_PATCH_SIZE 4

// The output patch size.  In this sample, it is also 3 vertices(control points).
#define OUTPUT_PATCH_SIZE 4


//--------------------------------------------------------------------------------------
// Vertex shader section
//--------------------------------------------------------------------------------------
struct VS_CONTROL_POINT_INPUT
{
    float4 vPosition        : SV_Position;
		float3 color : COLOR;

};

struct VS_CONTROL_POINT_OUTPUT
{
    float4 vPosition        : SV_Position;
		float3 pos : POSITION;

		float3 color : COLOR;
};

//----

cbuffer Locals {
	float4x4 u_Model;
	float4x4 u_View;
	float4x4 u_Proj;
	float4 u_ipos;
	float4 u_cpos;
	float4 u_color;
	float u_iscale;
};

struct VsOutput {
    float4 pos: SV_Position;
    float3 color: COLOR;
		float3 prepos: WORLDPOS;
};


float4x4 build_transform(float3 pos, float3 ang)
{
  float cosX = cos(ang.x);
  float sinX = sin(ang.x);
  float cosY = cos(ang.y);
  float sinY = sin(ang.y);
  float cosZ = cos(ang.z);
  float sinZ = sin(ang.z);

  float4x4 m;

  float m00 = cosY * cosZ + sinX * sinY * sinZ;
  float m01 = cosY * sinZ - sinX * sinY * cosZ;
  float m02 = cosX * sinY;
  float m03 = 0.0;

  float m04 = -cosX * sinZ;
  float m05 = cosX * cosZ;
  float m06 = sinX;
  float m07 = 0.0;

  float m08 = sinX * cosY * sinZ - sinY * cosZ;
  float m09 = -sinY * sinZ - sinX * cosY * cosZ;
  float m10 = cosX * cosY;
  float m11 = 0.0;

  float m12 = pos.x;
  float m13 = pos.y;
  float m14 = pos.z;
  float m15 = 1.0;

  /*
  //------ Orientation ---------------------------------
  m[0] = vec4(m00, m01, m02, m03); // first column.
  m[1] = vec4(m04, m05, m06, m07); // second column.
  m[2] = vec4(m08, m09, m10, m11); // third column.

  //------ Position ------------------------------------
  m[3] = vec4(m12, m13, m14, m15); // fourth column.
  */

  //------ Orientation ---------------------------------
  m[0][0] = m00; // first entry of the first column.
  m[0][1] = m01; // second entry of the first column.
  m[0][2] = m02;
  m[0][3] = m03;

  m[1][0] = m04; // first entry of the second column.
  m[1][1] = m05; // second entry of the second column.
  m[1][2] = m06;
  m[1][3] = m07;

  m[2][0] = m08; // first entry of the third column.
  m[2][1] = m09; // second entry of the third column.
  m[2][2] = m10;
  m[2][3] = m11;

  //------ Position ------------------------------------
  m[3][0] = m12; // first entry of the fourth column.
  m[3][1] = m13; // second entry of the fourth column.
  m[3][2] = m14;
  m[3][3] = m15;

  return m;
}




VS_CONTROL_POINT_OUTPUT Vertex(float3 pos : a_Pos, float3 color : a_Color) {
    float3 preposa = u_ipos+(pos);
    float3 prepos = u_ipos+(pos*u_iscale);
    float minlength = 25.0/256.0;
	//	float nscale = 0.009;
	float nscale = 0.002;

    //prepos.z = snoise(prepos.xy*0.05);
    //prepos.z = snoise(prepos.xy*0.05);
	//	float theight =  snoise(prepos.xy*nscale);



    float md = pos.x+pos.y;

    //prepos.z = prepos.z * 5.0;




/*
			    if (u_iscale < 1.0 && (pos.x >= 25.0 || pos.x <= -25.0))

			        float b = prepos.y/(minlength*u_iscale);
			        float a = floor(b*u_iscale);
			        float c = ceil(b*u_iscale);
			        float a1 = a*minlength;
			        float c1 = c*minlength;
			        float nb =snoise(float2(prepos.x,a1).xy*nscale);
			        float na =snoise(float2(prepos.x, c1).xy*nscale);
			        //prepos.z = (nb+na)/2.0;
							theight = (nb+na)/2.0;
			        color = float3(1.0, 0.0, 0.0);
			    }*/



		float radi = 25.0*100.0*2.0;
		float3 cofg = float3(0.0,0.0,-radi);
		float3 spherevec = normalize((prepos)-cofg);
		float3 prepos3 =  ( radi*spherevec)+cofg;


		//prepos3 = prepos+float3(0.0, 0.0,theight*100.0);
	prepos3 = prepos3 - u_ipos;
	//float4 inner = mul(u_Model, float4(prepos3, 1.0));
	float3 inner = mul(u_Model, float4(prepos3, 1.0));

	float ninner = inner.xyz;
		//float theight2 =  hybridTerrain(inner.xyz*nscale/20.0);
		float theight =  1.0;//hybridTerrain(inner.xyz*nscale*100.0, 64.0);


			if (theight > 0.4)
				color = lerp(float3(0.2, 0.7, 0.2), float3(0.9, 0.9, 0.9), (theight-0.8)/0.2); // white
			else if (theight > 0.2)
				color = float3(0.2, 0.7, 0.2); // green
			else
					color = float3(0.2, 0.2, 0.7); // blue*/
			//inner = inner + (float4(spherevec, 1.0)*theight*200.0);
			inner = (normalize(inner) * radi) + (normalize(inner) *theight*50.0);
				// IT WORKS float4 p = mul(u_Proj, mul(u_View, float4(inner, 1.0)));
				float4 p = float4(inner, 1.0);
		//color = u_color;
    VS_CONTROL_POINT_OUTPUT output;
		output.vPosition=  p;
		output.pos = inner;
		///output.color = color;
		//output.prepos = inner;
    return output;
}

//--------------------------------------------------------------------------------------
// Evaluation domain shader section
//--------------------------------------------------------------------------------------
struct DS_OUTPUT
{
    float4 vPosition        : SV_Position;
		float3 vPosition2 : WORLDPOS;
		float3 vNormal : POSITION;
		float3 color : COLOR;
};
float4 Pixel(DS_OUTPUT pin) : SV_Target {

	float nscale = 1.5;
	float shift = 0.0001;
	//Interpolation to find each position the generated vertices
	float3 finalPos = pin.vPosition2;

	/*
	float3 finalPosPolarB = cart2pol(finalPos);
	finalPosPolarB.y += shift;
	float3 finalPosB = pol2cart(swap_yz(finalPosPolarB));

	float3 finalPosPolarC = cart2pol(finalPos);
	finalPosPolarC.z += shift;
	float3 finalPosC = pol2cart(swap_yz(finalPosPolarC));
	*/

	float3 finalPosB = mul(build_transform(float3(0.0,0.0,0.0), float3(shift, 0.0, 0.0)),finalPos) ;
	float3 finalPosC = mul(build_transform(float3(0.0,0.0,0.0), float3(0.0, shift, 0.0)), finalPos);



	float4 finalPos2 = mul(u_Proj, mul(u_View, float4(finalPos, 1.0)));




	float radi = 25.0*100.0*2.0;
	float3 cofg = float3(0.0,0.0,-radi);
	float3 spherevec = normalize((finalPos)-cofg);
	float3 prepos3 =  ( radi*spherevec)+cofg;


	//float4 inner = mul(u_Model, float4(prepos3, 1.0));
	float3 inner = finalPos;

	//theightX = 1.0 - theightX;
	float theight =  (turbulence(finalPos*nscale, 2.0));
		float theight1 =  (turbulence(finalPosB*nscale, 2.0));
			float theight2 =  (turbulence(finalPosC*nscale, 2.0));



		//	theight = theight*theight;
		//	theight1 = theight1*theight1;
		//	theight2 = theight2*theight2;

		//	hnorm = mul(u_Model, hnorm);
	//theight = pow(theight , 2.1);

			float3 hnorm = normalize(float3(theight - theight1, 1.0, theight - theight2 ));

	//	if (theight < 0.0)
		//		color = float3(1.0, 0.0, 0.0); // reed
		//inner = inner + (float4(spherevec, 1.0)*theight*200.0);

	inner = (normalize(inner) * radi) + (normalize(inner) *theight*50.0);
			// IT WORKS float4 p = mul(u_Proj, mul(u_View, float4(inner, 1.0)));
	//Output.vNormal = normalize(mul(u_Proj, mul(u_View, float4(normalize(inner), 1.0))));
	//Output.vPosition = mul( float4(finalPos,1), mul(u_View, u_Proj) );
	//Output.vPosition =mul(u_Proj, mul(u_View, float4(finalPos, 1.0)));

	//    Output.vPosition = mul( float4(finalPos,1), (u_View) );
	finalPos2 = mul(u_Proj, mul(u_View, float4(inner, 1.0)));

	float4x4 modelMatrix = u_Model;
	float4x4 modelMatrixInverse = rcp(u_Model);

	float3 normalDirection = lerp(pin.vNormal,hnorm, 0.3);

	 float3 _WorldSpaceLightPos0 = normalize(float3(1.0, 0.0, 1.0));
	float3 lightDirection = normalize(_WorldSpaceLightPos0.xyz);

	float diffuseReflection = max(0.04, dot(normalDirection, lightDirection));


//	Output.color = lerp(color, color*diffuseReflection, 1.0);
//	float4 c = float4(theight2,theight2,theight2,1.0);
    return float4(lerp(pin.color, pin.color.xyz*diffuseReflection, 0.4), 1.0);
}


// This allows us to compile the shader with a #define to choose
// the different partition modes for the hull shader.
// See the hull shader: [partitioning(BEZIER_HS_PARTITION)]
// This sample demonstrates "integer", "fractional_even", and "fractional_odd"
#ifndef HS_PARTITION
#define HS_PARTITION "integer"
#endif //HS_PARTITION

// The input patch size.  In this sample, it is 3 vertices(control points).
// This value should match the call to IASetPrimitiveTopology()
#define INPUT_PATCH_SIZE 4

// The output patch size.  In this sample, it is also 3 vertices(control points).
#define OUTPUT_PATCH_SIZE 4

//----------------------------------------------------------------------------------
// Constant data function for the HS.  This is executed once per patch.
//--------------------------------------------------------------------------------------
struct HS_CONSTANT_DATA_OUTPUT
{
    float Edges[4]             : SV_TessFactor;
    float Inside [2]          : SV_InsideTessFactor;
};

struct HS_OUTPUT
{
    float3 vPosition           : POSITION;

};

float alerp(float a, float b, float c)
{
	return (c-a) / (b-a);
}
HS_CONSTANT_DATA_OUTPUT ConstantHS( InputPatch<VS_CONTROL_POINT_OUTPUT, 4> ip,
                                          uint PatchID : SV_PrimitiveID)
{
	float max_tess = 16.0;
	float min_tess = 1.0;

  float radiiii = 5000.0;
	//float3 campos = float3(u_View[3][0],u_View[3][1],u_View[3][2]);
	float3 campos = u_cpos.xyz;

	//float vv = max(0.0, min(alerp(radiiii, radiiii*2.0,length(campos)), 1.0)) ;
	float3 vpos = (ip[0].pos+ip[1].pos+ip[2].pos+ip[3].pos)/4.0;
	float vv = max(0.0, min(alerp(0.0, radiiii/4.0,length(vpos+campos)), 1.0)) ;
	//vv = vv*vv*vv*vv;
	float finaltess = lerp(min_tess, max_tess, (1.0-vv)*(1.0-vv)*(1.0-vv)*(1.0-vv));

	float g_fTessellationFactor = finaltess;//16.0;//8.0
    HS_CONSTANT_DATA_OUTPUT Output;

    Output.Edges[0] = Output.Edges[1] = Output.Edges[2] = Output.Edges[3] = g_fTessellationFactor;
    Output.Inside [0] = Output.Inside [1] = g_fTessellationFactor;

    return Output;
}

// The hull shader is called once per output control point, which is specified with
// outputcontrolpoints.  For this sample, we take the control points from the vertex
// shader and pass them directly off to the domain shader.  In a more complex scene,
// you might perform a basis conversion from the input control points into a Bezier
// patch, such as the SubD11 Sample of DirectX SDK.

// The input to the hull shader comes from the vertex shader

// The output from the hull shader will go to the domain shader.
// The tessellation factor, topology, and partition mode will go to the fixed function
// tessellator stage to calculate the UV and domain points.

[domain("quad")] //Quad domain for our shader
[partitioning(HS_PARTITION)] //Partitioning type according to the GUI
[outputtopology("triangle_cw")] //Where the generated triangles should face
[outputcontrolpoints(4)] //Number of times this part of the hull shader will be called for each patch
[patchconstantfunc("ConstantHS")] //The constant hull shader function
HS_OUTPUT HS( InputPatch<VS_CONTROL_POINT_OUTPUT, 4> p,
                    uint i : SV_OutputControlPointID,
                    uint PatchID : SV_PrimitiveID )
{
    HS_OUTPUT Output;
    Output.vPosition = p[i].vPosition;
/*
		Output.vPosition1 = p[0].vPosition;
		Output.vPosition2 = p[1].vPosition;
		Output.vPosition3 = p[2].vPosition;
		Output.vPosition4 = p[3].vPosition;*/
    return Output;
}


float3 swap_yz(float3 i)
{
		float y = i.y;
		return float3(i.x, i.z, y);
}
float3 cart2pol(float3 p)
{
    float r = sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
    float theta = atan2(p.y, p.x);
		float phi = atan2((sqrt(p.x*p.x + p.y*p.y)),p.z);
    return float3(r, theta, phi);
}
float3 pol2cart(float3 o)
{
		float3 z = float3(o.x * cos(o.y)* sin(o.z),
										  o.x * sin(o.y)*sin(o.z),
											o.x * cos(o.z));
    return z;
}

//Domain Shader is invoked for each vertex created by the Tessellator
[domain("quad")]
DS_OUTPUT DS( HS_CONSTANT_DATA_OUTPUT input,
                    float2 UV : SV_DomainLocation,
                    const OutputPatch<HS_OUTPUT, 4> quad )
{
  DS_OUTPUT Output;
		float nscaleO = 0.002;//0.002;
	float nscale = 0.006;//0.002;
	float shift = 0.01;
	//Interpolation to find each position the generated vertices
	float3 verticalPos1 = lerp(quad[0].vPosition,quad[1].vPosition,UV.y);
	float3 verticalPos2 = lerp(quad[3].vPosition,quad[2].vPosition,UV.y);
	float3 finalPos = lerp(verticalPos1,verticalPos2,UV.x);

	/*
	float3 finalPosPolarB = cart2pol(finalPos);
	finalPosPolarB.y += shift;
	float3 finalPosB = pol2cart(swap_yz(finalPosPolarB));

	float3 finalPosPolarC = cart2pol(finalPos);
	finalPosPolarC.z += shift;
	float3 finalPosC = pol2cart(swap_yz(finalPosPolarC));
	*/

float3 finalPosB = mul(build_transform(float3(0.0,0.0,0.0), float3(shift, 0.0, 0.0)),finalPos) ;
float3 finalPosC = mul(build_transform(float3(0.0,0.0,0.0), float3(0.0, shift, 0.0)), finalPos);



	float4 finalPos2 = mul(u_Proj, mul(u_View, float4(finalPos, 1.0)));




	float radi = 25.0*100.0*2.0;
	float3 cofg = float3(0.0,0.0,-radi);
	float3 spherevec = normalize((finalPos)-cofg);
	float3 prepos3 =  ( radi*spherevec)+cofg;


//float4 inner = mul(u_Model, float4(prepos3, 1.0));
float3 inner = finalPos;

	// IT WORKS float4 p = mul(u_Proj, mul(u_View, float4(inner, 1.0)));
	float theightX =  snoise(finalPos*nscaleO/9.0);//, 19.0);
if (theightX < 0.3)
		theightX = -0.7;
	else
	theightX = 0.3;
	//theightX = 1.0 - theightX;
	float theight =  (theightX+turbulence(finalPos*nscale, 32.0));
		float theight1 =  (theightX+turbulence(finalPosB*nscale, 32.0));
			float theight2 =  (theightX+turbulence(finalPosC*nscale, 32.0));



			theight = theight*theight;
			theight1 = theight1*theight1;
				theight2 = theight2*theight2;

		//	hnorm = mul(u_Model, hnorm);
//theight = pow(theight , 2.1);

		float3 color = float3(0.0,0.0,0.0);
		if (theight > 1.4)
			color = float3(0.9, 0.9, 0.9); // white
		else if (theight > 1.3)
				color = float3(0.7, 0.7, 0.7); // greay
		else
		 if (theight > 1.2)
				color = float3(0.678431373, 0.360784314, 0.164705882); // greay
		else if (theight > 0.4)
			color = float3(0.2, 0.7, 0.2); // green
		else
		{
				color = float3(0.2, 0.2, 0.7); // blue
					theight = 0.4;
					theight1 = 0.4;
					theight2 = 0.4;
		}
	color = color* (theight);
	if (theight < 0.4)
			color = float3(0.2, 0.2, 0.7); // blue
			float3 hnorm = normalize(float3(theight - theight1, 1.0, theight - theight2 ));

	//	if (theight < 0.0)
		//		color = float3(1.0, 0.0, 0.0); // reed
		//inner = inner + (float4(spherevec, 1.0)*theight*200.0);

	inner = (normalize(inner) * radi) + (normalize(inner) *theight*10.0); //50.0
			// IT WORKS float4 p = mul(u_Proj, mul(u_View, float4(inner, 1.0)));
//Output.vNormal = normalize(mul(u_Proj, mul(u_View, float4(normalize(inner), 1.0))));
Output.vNormal = lerp(-hnorm, normalize(inner), 0.6	);
//Output.vPosition = mul( float4(finalPos,1), mul(u_View, u_Proj) );
//Output.vPosition =mul(u_Proj, mul(u_View, float4(finalPos, 1.0)));

//    Output.vPosition = mul( float4(finalPos,1), (u_View) );
finalPos2 = mul(u_Proj, mul(u_View, float4(inner, 1.0)));

Output.vPosition = finalPos2;
Output.vPosition2 = finalPos;

float4x4 modelMatrix = u_Model;
float4x4 modelMatrixInverse = rcp(u_Model);

float3 normalDirection = normalize(
	 mul(float4(-Output.vNormal, 0.0), modelMatrixInverse).xyz);

	 normalDirection = Output.vNormal;

	 float3 _WorldSpaceLightPos0 = normalize(float3(1.0, 0.0, 1.0));
float3 lightDirection = normalize(_WorldSpaceLightPos0.xyz);

float diffuseReflection = max(0.04, dot(normalDirection, lightDirection));


Output.color = lerp(color, color*diffuseReflection, 1.0);
    return Output;
}
